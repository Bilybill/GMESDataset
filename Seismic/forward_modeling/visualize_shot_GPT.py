#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visualize_shot.py

Deepwave shot gather visualizer with 3 switchable modes.

Modes
-----
--mode inline
    Extract a single receiver line (y ~= shot_y) from a 2D receiver grid and
    plot a classic 2D-style gather (time vs inline offset).

--mode offset_binned --bin_m 40
    Compute horizontal offsets in meters, bin receivers by offset (uniform bin
    width in meters), stack within each bin (mean), then plot time vs binned
    offset. This avoids the common artifact where non-uniform offsets are
    displayed on a uniform imshow axis.

--mode grid_slice --t 0.8
    Reshape receivers back to a 2D (y,x) grid and plot an amplitude map on the
    receiver plane at a fixed time t (seconds).

Input
-----
Expects an .npz produced by your forward code, containing:
    data: [S, R, T]
    dt:   scalar
    src:  [S, nsrc, dim]  (dim=2 or 3)
    rec:  [S, R, dim]
    dx:   [dim]           (grid spacings)

Notes
-----
- For 3D, this script uses *horizontal offset* (x,y) by default (ignores z) which
  matches the usual seismic definition.
- Geometry arrays in your pipeline are stored in *grid indices*. We multiply by
  dx to get meters.

Examples
--------
python visualize_shot.py --file shot_gathers_3D.npz --mode inline
python visualize_shot.py --file shot_gathers_3D.npz --mode offset_binned --bin_m 40
python visualize_shot.py --file shot_gathers_3D.npz --mode grid_slice --t 0.8

"""

import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def _as_float(x) -> float:
    arr = np.array(x).reshape(-1)
    return float(arr[0])


def _clip_symmetric(x: np.ndarray, pct: float) -> float:
    """Symmetric percentile clip based on |x|."""
    pct = float(np.clip(pct, 50.0, 100.0))
    vmax = np.percentile(np.abs(x), pct)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(x)) + 1e-12)
    return float(vmax)


def _compute_offset_horizontal_m(src0: np.ndarray, rec: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Horizontal offset in meters.

    src0: [dim]
    rec : [R, dim]
    dx  : [dim]

    For dim==2: uses x only (classic 2D line). For dim>=3: uses (x,y).
    """
    dx = np.asarray(dx, dtype=np.float64).reshape(-1)
    src0 = np.asarray(src0, dtype=np.float64).reshape(-1)
    rec = np.asarray(rec, dtype=np.float64)

    dim = src0.shape[0]
    d = (rec - src0) * dx[:dim]

    if dim == 2:
        return np.abs(d[:, 0])
    # dim >= 3: horizontal (x,y)
    return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)


def _compute_inline_offset_m(src0: np.ndarray, rec: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Signed inline offset in meters using x only."""
    dx = np.asarray(dx, dtype=np.float64).reshape(-1)
    src0 = np.asarray(src0, dtype=np.float64).reshape(-1)
    rec = np.asarray(rec, dtype=np.float64)

    d = (rec - src0) * dx[: src0.shape[0]]
    return d[:, 0]


def _infer_receiver_grid(rec_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Infer unique sorted y and x coordinates (grid indices) from rec positions."""
    x = rec_xy[:, 0]
    y = rec_xy[:, 1]
    ux = np.unique(x)
    uy = np.unique(y)
    ux.sort()
    uy.sort()
    return uy, ux


def _grid_map_values(
    rec_xy: np.ndarray,
    values: np.ndarray,
    uy: np.ndarray,
    ux: np.ndarray,
) -> np.ndarray:
    """Map receiver values onto a (ny,nx) grid using (y,x) indexing.

    rec_xy: [R,2] in grid indices
    values: [R]
    uy/ux: unique sorted y and x coordinates
    """
    ny, nx = len(uy), len(ux)
    grid = np.full((ny, nx), np.nan, dtype=np.float64)

    yi = np.searchsorted(uy, rec_xy[:, 1])
    xi = np.searchsorted(ux, rec_xy[:, 0])

    sum_grid = np.zeros((ny, nx), dtype=np.float64)
    cnt_grid = np.zeros((ny, nx), dtype=np.int64)

    for r in range(values.shape[0]):
        yk = yi[r]
        xk = xi[r]
        if 0 <= yk < ny and 0 <= xk < nx:
            v = values[r]
            if np.isfinite(v):
                sum_grid[yk, xk] += float(v)
                cnt_grid[yk, xk] += 1

    mask = cnt_grid > 0
    grid[mask] = sum_grid[mask] / cnt_grid[mask]
    return grid


def _resolve_output_path(out_path: str, in_file: str, mode: str, shot_idx: int) -> str:
    """Resolve output path; if directory provided, create a suitable filename."""
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        base = os.path.splitext(os.path.basename(in_file))[0]
        return os.path.join(out_path, f"{base}_{mode}_shot{shot_idx}.png")

    parent = os.path.dirname(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Deepwave shot gather visualizer (3 modes)")
    parser.add_argument("--file", required=True, help="Path to .npz file")
    parser.add_argument("--shot", type=int, default=0, help="Initial shot index")

    parser.add_argument(
        "--mode",
        type=str,
        default="inline",
        choices=["inline", "offset_binned", "grid_slice"],
        help="Visualization mode",
    )

    # mode-specific
    parser.add_argument("--bin_m", type=float, default=40.0, help="Offset bin width (m) for offset_binned")
    parser.add_argument("--t", type=float, default=0.8, help="Time (s) for grid_slice")

    parser.add_argument("--clip_pct", type=float, default=99.0, help="Percentile for clipping (default 99)")
    parser.add_argument(
        "--output",
        help=(
            "Path to save the visualization. If a directory (or ends with /), "
            "auto-generates a filename that includes mode and shot index."
        ),
    )
    args = parser.parse_args()

    # Load
    try:
        npz = np.load(args.file, allow_pickle=True)
        data = npz["data"]  # [S, R, T]
        dt = _as_float(npz["dt"])

        if "src" in npz and "rec" in npz and "dx" in npz:
            src = npz["src"]
            rec = npz["rec"]
            dx = np.array(npz["dx"]).reshape(-1)
            has_geom = True
        else:
            src, rec, dx = None, None, None
            has_geom = False
            print("[WARN] Geometry (src/rec/dx) not found. Some modes may not work.")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if data.ndim != 3:
        print(f"Error: expected data shape [Shots, Receivers, Time], got {data.shape}")
        return

    S, R, T = data.shape
    t_axis = np.arange(T, dtype=np.float64) * dt
    t_max = t_axis[-1]

    init_shot = int(np.clip(args.shot, 0, S - 1))

    print(f"Loaded: {args.file}")
    print(f"Data shape: {data.shape} (Shots={S}, Receivers={R}, TimeSteps={T})")
    print(f"dt = {dt} s (t_max ≈ {t_max:.3f} s)")
    print(f"mode = {args.mode}")

    if args.mode != "inline" and not has_geom:
        print("Error: selected mode requires geometry (src/rec/dx) but it is missing.")
        return

    # --- Mode implementations -------------------------------------------------
    def make_inline_gather(idx: int):
        g = data[idx]  # [R,T]
        s0 = src[idx, 0, :]
        r = rec[idx]

        y_shot = float(s0[1]) if s0.shape[0] >= 2 else 0.0
        y_vals = r[:, 1]
        y0 = y_vals[np.argmin(np.abs(y_vals - y_shot))]
        mask = (y_vals == y0)
        if mask.sum() < 2:
            k = min(256, R)
            order_y = np.argsort(np.abs(y_vals - y_shot))[:k]
            mask = np.zeros(R, dtype=bool)
            mask[order_y] = True
            y0 = y_vals[order_y[0]]

        g_line = g[mask]  # [N,T]
        r_line = r[mask]

        x_off = _compute_inline_offset_m(s0, r_line, dx)  # signed
        order = np.argsort(x_off)

        gather = g_line[order].T  # [T,N]
        x_m = x_off[order]
        extent = (float(x_m[0]), float(x_m[-1]), float(t_max), 0.0)
        xlabel = "Inline offset (m)"
        title = f"Inline gather (y≈{float(y0):.3f} index) | Shot {idx}"
        return gather, extent, xlabel, title

    def make_offset_binned(idx: int):
        g = data[idx]  # [R,T]
        s0 = src[idx, 0, :]
        r = rec[idx]

        offsets = _compute_offset_horizontal_m(s0, r, dx)  # [R] meters
        max_off = float(np.max(offsets))
        bin_w = float(max(args.bin_m, 1e-6))

        nb = int(np.floor(max_off / bin_w)) + 1
        edges = np.linspace(0.0, nb * bin_w, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        bin_id = np.digitize(offsets, edges) - 1
        bin_id = np.clip(bin_id, 0, nb - 1)

        out = np.full((nb, T), np.nan, dtype=np.float64)
        for b in range(nb):
            m = (bin_id == b)
            if not np.any(m):
                continue
            out[b] = np.mean(g[m].astype(np.float64), axis=0)

        gather = out.T  # [T,NB]

        empty = ~np.isfinite(out[:, 0])
        if np.any(empty):
            gather = np.ma.array(gather, mask=np.tile(empty[None, :], (T, 1)))

        extent = (float(centers[0]), float(centers[-1]), float(t_max), 0.0)
        xlabel = f"Offset (m), binned (Δ={bin_w:g} m)"
        title = f"Offset-binned gather | Shot {idx}"
        return gather, extent, xlabel, title

    def make_grid_slice(idx: int, t_sec: float):
        g = data[idx]  # [R,T]
        s0 = src[idx, 0, :]
        r = rec[idx]

        if s0.shape[0] < 3:
            raise ValueError("grid_slice expects 3D geometry (dim=3).")

        it = int(np.clip(np.round(t_sec / dt), 0, T - 1))
        t_used = float(t_axis[it])

        rec_xy = r[:, :2]
        uy, ux = _infer_receiver_grid(rec_xy)
        ny, nx = len(uy), len(ux)
        if ny * nx != R:
            print(
                f"[WARN] Cannot form a full (ny,nx) grid because ny*nx={ny*nx} != R={R}. "
                "Will still map values to (y,x) unique coordinates (missing cells become NaN)."
            )

        vals = g[:, it].astype(np.float64)
        grid = _grid_map_values(rec_xy, vals, uy, ux)  # [ny,nx]

        x_m = ux.astype(np.float64) * float(dx[0])
        y_m = uy.astype(np.float64) * float(dx[1])

        extent = (float(x_m[0]), float(x_m[-1]), float(y_m[-1]), float(y_m[0]))
        xlabel = "Receiver plane (x,y)"
        title = f"Receiver-plane slice @ t={t_used:.3f}s | Shot {idx}"
        return grid, extent, xlabel, title

    # --- Plotting -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    if args.mode == "inline":
        gather0, extent0, xlabel0, title0 = make_inline_gather(init_shot)
        v0 = _clip_symmetric(np.asarray(gather0), args.clip_pct)
        im = ax.imshow(gather0, aspect="auto", origin="upper", extent=extent0, cmap="seismic", vmin=-v0, vmax=v0)
        ax.set_ylabel("Time (s)")
        ax.set_xlabel(xlabel0)

    elif args.mode == "offset_binned":
        gather0, extent0, xlabel0, title0 = make_offset_binned(init_shot)
        v0 = _clip_symmetric(np.asarray(gather0), args.clip_pct)
        im = ax.imshow(gather0, aspect="auto", origin="upper", extent=extent0, cmap="seismic", vmin=-v0, vmax=v0)
        ax.set_ylabel("Time (s)")
        ax.set_xlabel(xlabel0)

    else:
        grid0, extent0, xlabel0, title0 = make_grid_slice(init_shot, args.t)
        finite = np.isfinite(grid0)
        v0 = _clip_symmetric(grid0[finite], args.clip_pct) if np.any(finite) else 1.0
        im = ax.imshow(grid0, aspect="auto", origin="upper", extent=extent0, cmap="seismic", vmin=-v0, vmax=v0)
        ax.set_ylabel("y (m)")
        ax.set_xlabel("x (m)")

    ax.set_title(title0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amplitude")

    # --- Sliders --------------------------------------------------------------
    ax_shot = plt.axes([0.15, 0.10, 0.70, 0.03])
    sld_shot = Slider(ax=ax_shot, label="Shot", valmin=0, valmax=S - 1, valinit=init_shot, valstep=1)

    sld_time = None
    if args.mode == "grid_slice":
        ax_time = plt.axes([0.15, 0.05, 0.70, 0.03])
        t0 = float(np.clip(args.t, 0.0, t_max))
        sld_time = Slider(ax=ax_time, label="t (s)", valmin=0.0, valmax=float(t_max), valinit=t0, valstep=float(dt))

    def _update(_):
        idx = int(sld_shot.val)

        if args.mode == "inline":
            g, ext, xl, ttl = make_inline_gather(idx)
            v = _clip_symmetric(np.asarray(g), args.clip_pct)
            im.set_data(g)
            im.set_extent(ext)
            im.set_clim(-v, v)
            ax.set_xlabel(xl)
            ax.set_ylabel("Time (s)")
            ax.set_title(ttl)

        elif args.mode == "offset_binned":
            g, ext, xl, ttl = make_offset_binned(idx)
            v = _clip_symmetric(np.asarray(g), args.clip_pct)
            im.set_data(g)
            im.set_extent(ext)
            im.set_clim(-v, v)
            ax.set_xlabel(xl)
            ax.set_ylabel("Time (s)")
            ax.set_title(ttl)

        else:
            t_now = float(sld_time.val) if sld_time is not None else float(args.t)
            grid, ext, _, ttl = make_grid_slice(idx, t_now)
            finite = np.isfinite(grid)
            v = _clip_symmetric(grid[finite], args.clip_pct) if np.any(finite) else 1.0
            im.set_data(grid)
            im.set_extent(ext)
            im.set_clim(-v, v)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(ttl)

        fig.canvas.draw_idle()

    sld_shot.on_changed(_update)
    if sld_time is not None:
        sld_time.on_changed(_update)

    # --- Save if requested ----------------------------------------------------
    if args.output:
        out_path = _resolve_output_path(args.output, args.file, args.mode, init_shot)
        plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.0)
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
