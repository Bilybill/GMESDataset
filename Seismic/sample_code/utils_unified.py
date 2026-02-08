import yaml
import torch
import numpy as np
import deepwave
import os


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_velocity_model(config: dict) -> torch.Tensor:
    """
    Loads velocity model from file or creates a dummy one if file doesn't exist.

    IMPORTANT:
      - This code assumes the velocity tensor dimension order matches your acquisition convention:
        2D: v.shape == [nx, nz]  (x is dim0, z is dim1)
        3D: v.shape == [nx, ny, nz] (x dim0, y dim1, z dim2)
    """
    path = config["model"]["file_path"]
    shape = config["model"]["shape"]

    if os.path.exists(path):
        if path.endswith(".npy"):
            v = np.load(path)
        elif path.endswith(".npz"):
            data = np.load(path)

            # Optional: composite model logic (kept compatible with your original utils.py)
            if "formatS" in data and "formatD" in data and "gtime" in data:
                print("Detected composite model components (formatS, formatD, gtime)...")
                Details = data["formatS"]
                MDetails = data["formatD"]
                Trends = data["gtime"]

                model_conf = config.get("model", {})
                trends_range = model_conf.get("trends_range", (2000, 6000))
                details_range = model_conf.get("details_range", (-500, 1000))
                mdetails_range = model_conf.get("mdetails_range", (-200, 200))

                # Normalize to [0,1] then scale to specified ranges
                def _scale(arr, vmin, vmax):
                    arr = arr.astype(np.float64)
                    if arr.max() != arr.min():
                        arr = (arr - arr.min()) / (arr.max() - arr.min())
                    return vmin + arr * (vmax - vmin)

                Trends = _scale(Trends, trends_range[0], trends_range[1])
                Details = _scale(Details, details_range[0], details_range[1])
                MDetails = _scale(MDetails, mdetails_range[0], mdetails_range[1])

                # Combine
                v = Trends + Details + MDetails
            else:
                # Fallback: try a common key
                if "v" in data:
                    v = data["v"]
                else:
                    raise ValueError(f"NPZ file {path} does not contain expected keys.")
        else:
            raise ValueError("Unsupported velocity file format. Use .npy or .npz")
    else:
        print(f"Velocity model file not found at {path}. Creating a dummy constant model for testing.")
        v = np.ones(shape, dtype=np.float32) * 2000.0  # Default 2000 m/s

    return torch.tensor(v, dtype=torch.float32)


def get_wavelet(config: dict, device: torch.device) -> torch.Tensor:
    """
    Generates source wavelet.
    """
    nt = config["time"]["nt"]
    dt = config["time"]["dt"]
    src_conf = config["source"]

    if src_conf.get("type", "ricker") == "ricker":
        freq = float(src_conf.get("freq", 25.0))
        peak_time = 1.5 / freq
        wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    else:
        print(f"Warning: Wavelet type {src_conf.get('type')} not implemented. Using Ricker.")
        freq = float(src_conf.get("freq", 25.0))
        peak_time = 1.5 / freq
        wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time)

    amp = float(src_conf.get("amplitude", 1.0))
    return (amp * wavelet).to(device)


# -----------------------
# Unified acquisition API
# -----------------------

def _resolve_y(y_value, ny: int) -> int:
    if y_value is None:
        return ny // 2
    if isinstance(y_value, str):
        if y_value.lower() == "center":
            return ny // 2
        # allow numeric strings
        if y_value.isdigit():
            return int(y_value)
        raise ValueError(f"Invalid y spec '{y_value}'. Use 'center' or an integer index.")
    return int(y_value)


def _oob_apply(locs: torch.Tensor, shape, policy: str) -> torch.Tensor:
    """
    shape: [nx, nz] for 2D, [nx, ny, nz] for 3D
    """
    policy = (policy or "error").lower()
    mins = locs.amin(dim=(0, 1))
    maxs = locs.amax(dim=(0, 1))

    ok = True
    for d, n in enumerate(shape):
        if mins[d].item() < 0 or maxs[d].item() >= n:
            ok = False
            break

    if ok:
        return locs

    if policy == "clamp":
        for d, n in enumerate(shape):
            locs[..., d] = locs[..., d].clamp(0, n - 1)
        return locs

    # error
    raise ValueError(
        f"Acquisition geometry out-of-bounds for shape={shape}. "
        f"mins={mins.tolist()}, maxs={maxs.tolist()}. "
        f"Set acquisition.oob_policy='clamp' to auto-clamp (not recommended for serious experiments)."
    )


def _get_shape_by_mode(config: dict):
    mode = config["simulation"]["mode"]
    shape = config["model"]["shape"]
    if mode == "2D":
        if len(shape) != 2:
            raise ValueError(f"2D mode expects model.shape length 2, got {shape}")
        nx, nz = int(shape[0]), int(shape[1])
        return (nx, nz)
    if mode == "3D":
        if len(shape) != 3:
            raise ValueError(f"3D mode expects model.shape length 3, got {shape}")
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        return (nx, ny, nz)
    raise ValueError(f"Unknown simulation.mode={mode}")


def setup_acquisition(config: dict, device: torch.device):
    """
    Backward compatible acquisition builder.

    Supported (unified):
      - Source layout: line (default/back-compat), grid, points
      - Receiver type: fixed (default/back-compat), rolling, patch
      - Receiver layout: line (default/back-compat), grid

    Returns:
      source_locations: [n_shots, n_src_per_shot, dim]
      receiver_locations: [n_shots, n_receivers_per_shot, dim]

    Side-effects:
      - If source.layout == grid or points, config['source']['n_shots'] is auto-filled (n_shots_x*n_shots_y or len(points))
      - If receiver.layout == grid (fixed/rolling/patch), config['receiver']['n_receivers_per_shot'] is auto-filled (flattened count)
      - For patch/rolling line, config['receiver']['n_receivers_per_shot'] is auto-filled if not provided
    """
    mode = config["simulation"]["mode"]
    dim = 2 if mode == "2D" else 3
    shape = _get_shape_by_mode(config)

    acq_conf = config.get("acquisition", {}) or {}
    oob_policy = acq_conf.get("oob_policy", "error")
    default_y = acq_conf.get("default_y", "center")

    s_conf = config.get("source", {}) or {}
    r_conf = config.get("receiver", {}) or {}

    # --------------
    # Build sources
    # --------------
    n_src_per_shot = int(s_conf.get("n_sources_per_shot", 1))
    s_layout = (s_conf.get("layout") or "line").lower()

    if s_layout == "line":
        n_shots = int(s_conf["n_shots"])
        first = int(s_conf["first_source"])
        ds = int(s_conf["d_source"])
        z_s = int(s_conf.get("source_depth", 2))

        sx = first + torch.arange(n_shots, device=device, dtype=torch.long) * ds

        if dim == 2:
            source_locations = torch.zeros((n_shots, n_src_per_shot, 2), device=device, dtype=torch.long)
            source_locations[:, :, 0] = sx.unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = z_s
        else:
            nx, ny, nz = shape
            y_s = _resolve_y(s_conf.get("source_y", default_y), ny)
            source_locations = torch.zeros((n_shots, n_src_per_shot, 3), device=device, dtype=torch.long)
            source_locations[:, :, 0] = sx.unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = y_s
            source_locations[:, :, 2] = z_s

    elif s_layout == "grid":
        # 3D areal shots; in 2D we interpret as x-only grid with n_shots_x
        z_s = int(s_conf.get("source_depth", 2))

        nsx = int(s_conf.get("n_shots_x", s_conf.get("n_shots", 1)))
        dsx = int(s_conf.get("d_source_x", s_conf.get("d_source", 1)))
        fsx = int(s_conf.get("first_source_x", s_conf.get("first_source", 0)))
        sx = fsx + torch.arange(nsx, device=device, dtype=torch.long) * dsx

        if dim == 2:
            n_shots = nsx
            config["source"]["n_shots"] = int(n_shots)
            source_locations = torch.zeros((n_shots, n_src_per_shot, 2), device=device, dtype=torch.long)
            source_locations[:, :, 0] = sx.unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = z_s
        else:
            nx, ny, nz = shape
            nsy = int(s_conf.get("n_shots_y", 1))
            dsy = int(s_conf.get("d_source_y", dsx))
            fsy = int(s_conf.get("first_source_y", ny // 2))
            sy = fsy + torch.arange(nsy, device=device, dtype=torch.long) * dsy

            XS, YS = torch.meshgrid(sx, sy, indexing="ij")
            shot_xy = torch.stack([XS.reshape(-1), YS.reshape(-1)], dim=-1)  # [n_shots, 2]
            n_shots = int(shot_xy.shape[0])

            config["source"]["n_shots"] = int(n_shots)

            source_locations = torch.zeros((n_shots, n_src_per_shot, 3), device=device, dtype=torch.long)
            source_locations[:, :, 0] = shot_xy[:, 0].unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 1] = shot_xy[:, 1].unsqueeze(1).repeat(1, n_src_per_shot)
            source_locations[:, :, 2] = z_s

    elif s_layout == "points":
        pts = s_conf.get("locations", None)
        if pts is None:
            raise ValueError("source.layout='points' requires source.locations list.")
        pts = torch.tensor(pts, device=device, dtype=torch.long)
        if dim == 2:
            # accept [x,z] or [x,y,z] (drop y)
            if pts.shape[1] == 3:
                pts2 = torch.stack([pts[:, 0], pts[:, 2]], dim=-1)
            elif pts.shape[1] == 2:
                pts2 = pts
            else:
                raise ValueError("2D source points must be [x,z] (2 cols) or [x,y,z] (3 cols).")
            n_shots = int(pts2.shape[0])
            config["source"]["n_shots"] = int(n_shots)
            source_locations = pts2.view(n_shots, 1, 2).repeat(1, n_src_per_shot, 1)
        else:
            nx, ny, nz = shape
            if pts.shape[1] == 2:
                # [x,z] + default y
                y_s = _resolve_y(s_conf.get("source_y", default_y), ny)
                pts3 = torch.stack([pts[:, 0], torch.full_like(pts[:, 0], y_s), pts[:, 1]], dim=-1)
            elif pts.shape[1] == 3:
                pts3 = pts
            else:
                raise ValueError("3D source points must be [x,y,z] (3 cols) or [x,z] (2 cols).")
            n_shots = int(pts3.shape[0])
            config["source"]["n_shots"] = int(n_shots)
            source_locations = pts3.view(n_shots, 1, 3).repeat(1, n_src_per_shot, 1)

    else:
        raise ValueError(f"Unknown source.layout={s_layout}")

    # Ensure in-bounds
    source_locations = _oob_apply(source_locations, shape, oob_policy)

    # ----------------
    # Build receivers
    # ----------------
    r_type = (r_conf.get("type") or "fixed").lower()
    r_layout = (r_conf.get("layout") or "line").lower()

    z_r = int(r_conf.get("receiver_depth", s_conf.get("source_depth", 2)))

    # Convenience: shot origins (x,y)
    shot_x = source_locations[:, 0, 0]  # [n_shots]
    shot_y = None
    if dim == 3:
        shot_y = source_locations[:, 0, 1]

    if r_type == "fixed":
        if r_layout == "line":
            n_rec = int(r_conf["n_receivers_per_shot"])
            first = int(r_conf["first_receiver"])
            dr = int(r_conf["d_receiver"])
            rx = first + torch.arange(n_rec, device=device, dtype=torch.long) * dr

            if dim == 2:
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rx.unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = z_r
            else:
                nx, ny, nz = shape
                y_r = _resolve_y(r_conf.get("receiver_y", default_y), ny)
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rx.unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = y_r
                receiver_locations[:, :, 2] = z_r

        elif r_layout == "grid":
            # fixed receiver grid, flattened
            if dim == 2:
                # interpret as line along x
                n_rec = int(r_conf.get("n_receivers_x", r_conf.get("n_receivers_per_shot")))
                first = int(r_conf.get("first_receiver_x", r_conf.get("first_receiver", 0)))
                dr = int(r_conf.get("d_receiver_x", r_conf.get("d_receiver", 1)))
                rx = first + torch.arange(n_rec, device=device, dtype=torch.long) * dr
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rx.unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = z_r
            else:
                nrx = int(r_conf["n_receivers_x"])
                nry = int(r_conf["n_receivers_y"])
                frx = int(r_conf["first_receiver_x"])
                fry = int(r_conf["first_receiver_y"])
                drx = int(r_conf["d_receiver_x"])
                dry = int(r_conf["d_receiver_y"])

                rx = frx + torch.arange(nrx, device=device, dtype=torch.long) * drx
                ry = fry + torch.arange(nry, device=device, dtype=torch.long) * dry
                RX, RY = torch.meshgrid(rx, ry, indexing="ij")
                rec_xy = torch.stack([RX.reshape(-1), RY.reshape(-1)], dim=-1)
                n_rec = int(rec_xy.shape[0])

                config["receiver"]["n_receivers_per_shot"] = int(n_rec)

                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = rec_xy[:, 0].unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 1] = rec_xy[:, 1].unsqueeze(0).repeat(source_locations.shape[0], 1)
                receiver_locations[:, :, 2] = z_r
        else:
            raise ValueError(f"Unknown receiver.layout={r_layout} for type=fixed")

    elif r_type == "rolling":
        # Receivers move with source
        if r_layout == "line":
            n_rec = int(r_conf["n_receivers_per_shot"])
            offset_first = int(r_conf.get("offset_first", r_conf.get("first_receiver", 0)))
            d_offset = int(r_conf.get("d_offset", r_conf.get("d_receiver", 1)))
            offx = offset_first + torch.arange(n_rec, device=device, dtype=torch.long) * d_offset  # [n_rec]

            if dim == 2:
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + offx.unsqueeze(0)
                receiver_locations[:, :, 1] = z_r
            else:
                y_off = int(r_conf.get("receiver_y_offset", 0))
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + offx.unsqueeze(0)
                receiver_locations[:, :, 1] = (shot_y + y_off).unsqueeze(1).repeat(1, n_rec)
                receiver_locations[:, :, 2] = z_r

        elif r_layout == "grid":
            if dim != 3:
                raise ValueError("receiver.type='rolling' with layout='grid' is only meaningful in 3D.")
            nrx = int(r_conf["n_receivers_x"])
            nry = int(r_conf["n_receivers_y"])
            offx0 = int(r_conf.get("offset_first_x", r_conf.get("first_receiver_x", 0)))
            offy0 = int(r_conf.get("offset_first_y", r_conf.get("first_receiver_y", 0)))
            doffx = int(r_conf.get("d_offset_x", r_conf.get("d_receiver_x", 1)))
            doffy = int(r_conf.get("d_offset_y", r_conf.get("d_receiver_y", 1)))

            offx = offx0 + torch.arange(nrx, device=device, dtype=torch.long) * doffx
            offy = offy0 + torch.arange(nry, device=device, dtype=torch.long) * doffy
            OFX, OFY = torch.meshgrid(offx, offy, indexing="ij")
            rel = torch.stack([OFX.reshape(-1), OFY.reshape(-1)], dim=-1)
            n_rec = int(rel.shape[0])
            config["receiver"]["n_receivers_per_shot"] = int(n_rec)

            receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
            receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + rel[:, 0].unsqueeze(0)
            receiver_locations[:, :, 1] = shot_y.unsqueeze(1) + rel[:, 1].unsqueeze(0)
            receiver_locations[:, :, 2] = z_r
        else:
            raise ValueError(f"Unknown receiver.layout={r_layout} for type=rolling")

    elif r_type == "patch":
        # Local receiver window around each shot.
        min_offset = int(r_conf.get("min_offset", 0))

        if r_layout == "line":
            half = int(r_conf.get("patch_half_span", 0))
            if half <= 0:
                raise ValueError("receiver.type='patch' layout='line' requires receiver.patch_half_span > 0")
            step = int(r_conf.get("patch_d", r_conf.get("d_receiver", 1)))
            relx = torch.arange(-half, half + 1, step, device=device, dtype=torch.long)

            if min_offset > 0:
                relx = relx[relx.abs() >= min_offset]

            n_rec = int(relx.numel())
            config["receiver"]["n_receivers_per_shot"] = int(n_rec)

            if dim == 2:
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 2), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + relx.unsqueeze(0)
                receiver_locations[:, :, 1] = z_r
            else:
                y_off = int(r_conf.get("receiver_y_offset", 0))
                receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
                receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + relx.unsqueeze(0)
                receiver_locations[:, :, 1] = (shot_y + y_off).unsqueeze(1).repeat(1, n_rec)
                receiver_locations[:, :, 2] = z_r

        elif r_layout == "grid":
            if dim != 3:
                raise ValueError("receiver.type='patch' with layout='grid' is only supported in 3D.")
            halfx = int(r_conf.get("patch_half_span_x", r_conf.get("patch_half_span", 0)))
            halfy = int(r_conf.get("patch_half_span_y", r_conf.get("patch_half_span", 0)))
            if halfx <= 0 or halfy <= 0:
                raise ValueError("patch grid requires patch_half_span_x > 0 and patch_half_span_y > 0")
            stepx = int(r_conf.get("patch_d_x", r_conf.get("patch_d", 1)))
            stepy = int(r_conf.get("patch_d_y", r_conf.get("patch_d", 1)))

            relx = torch.arange(-halfx, halfx + 1, stepx, device=device, dtype=torch.long)
            rely = torch.arange(-halfy, halfy + 1, stepy, device=device, dtype=torch.long)
            RX, RY = torch.meshgrid(relx, rely, indexing="ij")

            rel = torch.stack([RX.reshape(-1), RY.reshape(-1)], dim=-1)  # [N,2]

            # Optional circle mask in metric units
            mask = torch.ones((rel.shape[0],), device=device, dtype=torch.bool)
            patch_mask = (r_conf.get("patch_mask") or "box").lower()
            if patch_mask == "circle":
                dx = float(config["model"].get("dx", 1.0))
                dy = float(config["model"].get("dy", 1.0))
                radius_m = float(r_conf.get("patch_radius_m", 0.0))
                if radius_m <= 0:
                    raise ValueError("patch_mask='circle' requires receiver.patch_radius_m > 0")
                dist2 = (rel[:, 0].float() * dx) ** 2 + (rel[:, 1].float() * dy) ** 2
                mask = mask & (dist2 <= radius_m ** 2)

            if min_offset > 0:
                # min_offset in grid points (radial)
                r = torch.sqrt(rel[:, 0].float() ** 2 + rel[:, 1].float() ** 2)
                mask = mask & (r >= float(min_offset))

            rel = rel[mask]
            n_rec = int(rel.shape[0])
            config["receiver"]["n_receivers_per_shot"] = int(n_rec)

            receiver_locations = torch.zeros((source_locations.shape[0], n_rec, 3), device=device, dtype=torch.long)
            receiver_locations[:, :, 0] = shot_x.unsqueeze(1) + rel[:, 0].unsqueeze(0)
            receiver_locations[:, :, 1] = shot_y.unsqueeze(1) + rel[:, 1].unsqueeze(0)
            receiver_locations[:, :, 2] = z_r

        else:
            raise ValueError(f"Unknown receiver.layout={r_layout} for type=patch")

    else:
        raise ValueError(f"Unknown receiver.type={r_type}")

    receiver_locations = _oob_apply(receiver_locations, shape, oob_policy)
    return source_locations, receiver_locations
