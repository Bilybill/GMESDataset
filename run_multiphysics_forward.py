import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from core.multiphysics import build_multiphysics_model
from core.presets import FORWARD_ANOMALY_TYPES, build_named_anomaly_preset, read_segy_volume
from Seismic.forward_modeling.utils import get_wavelet, setup_acquisition

from core.forward_modeling.gravity import GravityForwardSolver
from core.forward_modeling.magnetic import MagneticForwardSolver
from core.forward_modeling.electrical import ElectricalForwardSolver
from core.forward_modeling.seismic import SeismicForwardSolver


DENSITY_UNIT = "g/cm^3"
SEISMIC_PRESETS = ("full", "light")

def _normalize_density_value_for_pipeline(value):
    value = float(value)
    if value > 50.0:
        return value / 1000.0
    return value


def _format_exception_message(exc):
    return " ".join(str(exc).strip().split())


def _resolve_torch_device(preferred_device="auto"):
    preferred = str(preferred_device).lower()
    if preferred not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device option: {preferred_device}")

    if preferred == "cpu":
        return torch.device("cpu"), "Torch device: CPU (requested by user)."

    if not torch.cuda.is_available():
        reason = "Torch device: CPU (CUDA not available)."
        if preferred == "cuda":
            reason = "Torch device: CPU (requested CUDA, but CUDA is not available)."
        return torch.device("cpu"), reason

    try:
        probe = torch.arange(4, dtype=torch.float32, device="cuda")
        _ = float((probe.square().sum()).item())
        torch.cuda.synchronize()
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        return torch.device("cuda"), f"Torch device: CUDA ({name}, sm_{capability[0]}{capability[1]})."
    except Exception as exc:
        reason = (
            "Torch device: CPU "
            f"(CUDA is visible but failed validation: {_format_exception_message(exc)})."
        )
        return torch.device("cpu"), reason


def _move_tensor(tensor, device):
    return tensor if tensor.device == device else tensor.to(device)


def _tensor_to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def _conductivity_domain_downsample(res_tensor, target_shape):
    sigma_tensor = torch.clamp(res_tensor, min=1.0e-3).reciprocal()
    sigma_5d = sigma_tensor.unsqueeze(0).unsqueeze(0)
    sigma_down = torch.nn.functional.interpolate(
        sigma_5d,
        size=target_shape,
        mode='trilinear',
        align_corners=True,
    ).squeeze().contiguous()
    return torch.clamp(sigma_down, min=1.0e-6).reciprocal()


def _apply_boundary_background_taper(res_tensor, edge_cells=4):
    nx, ny, nz = res_tensor.shape
    if min(nx, ny) <= edge_cells * 2:
        return res_tensor

    log_res = torch.log10(torch.clamp(res_tensor, min=1.0e-3))
    background_1d = torch.median(log_res.reshape(nx * ny, nz), dim=0).values
    background_3d = background_1d.view(1, 1, nz).expand(nx, ny, nz)

    wx = torch.ones(nx, dtype=log_res.dtype, device=log_res.device)
    wy = torch.ones(ny, dtype=log_res.dtype, device=log_res.device)
    taper = torch.linspace(0.0, 1.0, edge_cells + 1, dtype=log_res.dtype, device=log_res.device)[:-1]
    wx[:edge_cells] = taper
    wx[-edge_cells:] = torch.flip(taper, dims=[0])
    wy[:edge_cells] = taper
    wy[-edge_cells:] = torch.flip(taper, dims=[0])

    weight = torch.minimum(wx[:, None], wy[None, :]).unsqueeze(-1)
    blended = weight * log_res + (1.0 - weight) * background_3d
    return torch.pow(10.0, blended)


def _save_mt_xz_slice(res_tensor, spacing, save_path, log_scale=True):
    nx, ny, nz = res_tensor.shape
    dx, _, dz = spacing
    mid_y = ny // 2
    slice_xz = res_tensor[:, mid_y, :].detach().cpu().numpy()
    image = np.log10(slice_xz.T) if log_scale else slice_xz.T
    colorbar_label = "Log10 Resistivity (Ohm-m)" if log_scale else "Resistivity (Ohm-m)"
    title_suffix = "log10" if log_scale else "linear"

    plt.figure(figsize=(8, 5))
    plt.imshow(
        image,
        aspect="auto",
        cmap="jet",
        extent=[0, nx * dx, nz * dz, 0],
    )
    plt.colorbar(label=colorbar_label)
    plt.title(f"Downsampled MT Resistivity X-Z Slice ({title_suffix}, y={mid_y})")
    plt.xlabel("X (m)")
    plt.ylabel("Depth (m)")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[+] Saved MT resistivity X-Z slice to: {save_path}")


def _build_seismic_3d_config(shape, spacing, preset="full"):
    nx, ny, nz = shape
    dx, dy, dz = spacing

    preset = str(preset).lower()
    if preset not in SEISMIC_PRESETS:
        raise ValueError(f"Unsupported seismic preset: {preset}")

    if preset == "light":
        n_shots_x = min(3, nx)
        n_shots_y = min(3, ny)
        n_receivers_x = min(32, nx)
        n_receivers_y = min(32, ny)
    else:
        n_shots_x = min(5, nx)
        n_shots_y = min(5, ny)
        n_receivers_x = min(64, nx)
        n_receivers_y = min(64, ny)

    d_receiver_x = max(nx // max(n_receivers_x, 1), 1)
    d_receiver_y = max(ny // max(n_receivers_y, 1), 1)

    return {
        "simulation": {
            "mode": "3D",
            "accuracy": 4,
        },
        "acquisition": {
            "default_y": "center",
            "oob_policy": "error",
            "show_geometry": False,
        },
        "model": {
            "shape": [nx, ny, nz],
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
        },
        "time": {
            "nt": 750,
            "dt": 0.004,
        },
        "source": {
            "type": "ricker",
            "freq": 10.0,
            "amplitude": 1.0,
            "n_sources_per_shot": 1,
            "layout": "grid_random",
            "source_depth": 2,
            "n_shots_x": n_shots_x,
            "n_shots_y": n_shots_y,
        },
        "receiver": {
            "type": "fixed",
            "layout": "grid",
            "receiver_depth": 2,
            "n_receivers_x": n_receivers_x,
            "n_receivers_y": n_receivers_y,
            "first_receiver_x": 0,
            "first_receiver_y": 0,
            "d_receiver_x": d_receiver_x,
            "d_receiver_y": d_receiver_y,
        },
    }

def generate_model(vp_segy_path, label_segy_path, anomaly_type="igneous_swarm"):
    """读取 SEGY 构建指定异常体的四参数模型，用于四法联合正演基准。"""
    try:
        vp_bg, (dx, dy, dz) = read_segy_volume(vp_segy_path)
        label_vol, _ = read_segy_volume(label_segy_path)
        nx, ny, nz = vp_bg.shape
        print(f'Velocity shape = {nx, ny, nz}, dx={dx}, dy={dy}, dz={dz}')
    except Exception as e:
        raise RuntimeError(f"Failed to read SEGY volumes: {e}")

    preset = build_named_anomaly_preset(anomaly_type, vp_bg, label_vol, (dx, dy, dz))
    print(f"Selected anomaly: {preset.name_en} ({preset.name_zh}) [{preset.key}]")

    models = build_multiphysics_model(vp_bg, label_vol, [preset.anomaly], dx, dy, dz)
    print(f'背景密度 value range = {models["rho_bg"].min():.2f} - {models["rho_bg"].max():.2f} {DENSITY_UNIT}')
    print(f'背景电阻率 value range = {models["resist_bg"].min():.2e} - {models["resist_bg"].max():.2e} Ohm-m')

    return {
        "vp": models["vp"],
        "rho": models["rho"],
        "res": models["resist"],
        "chi": models["chi"],
        "rho_bg": models["rho_bg"],
        "chi_bg": models["chi_bg"],
        "anomaly_label": models["anomaly_label"],
        "shape": (nx, ny, nz),
        "spacing": (dx, dy, dz),
        "facies_bg": None if models["facies_bg"] is None else models["facies_bg"].astype(np.int16),
        "anomaly_type": preset.key,
        "anomaly_name_en": preset.name_en,
        "anomaly_name_zh": preset.name_zh,
    }

def run_forward_pipeline(save_dir, vp_segy_path, label_segy_path, run_gravity=True, run_magnetic=True, run_electrical=True, run_seismic=True, gravity_anomaly_mode="background", gravity_bg_density=2.67, torch_device_preference="auto", anomaly_type="igneous_swarm", seismic_preset="full"):
    print("====== 1. 模型生成 ======")
    gravity_bg_density = _normalize_density_value_for_pipeline(gravity_bg_density)
    model = generate_model(vp_segy_path, label_segy_path, anomaly_type=anomaly_type)
    vp_multi = model["vp"]
    rho_multi = model["rho"]
    res_multi = model["res"]
    chi_multi = model["chi"]
    rho_bg_arr = model["rho_bg"]
    chi_bg_arr = model["chi_bg"]
    facies_bg = model["facies_bg"]
    nx, ny, nz = model["shape"]
    dx, dy, dz = model["spacing"]
    print(
        f"Generated Models: {nx}x{ny}x{nz} (dx={dx}, dy={dy}, dz={dz}) "
        f"for {model['anomaly_name_en']} ({model['anomaly_name_zh']})."
    )

    bundle = {
        "vp_model": np.asarray(vp_multi, dtype=np.float32),
        "rho_model": np.asarray(rho_multi, dtype=np.float32),
        "res_model": np.asarray(res_multi, dtype=np.float32),
        "chi_model": np.asarray(chi_multi, dtype=np.float32),
        "rho_bg_model": np.asarray(rho_bg_arr, dtype=np.float32),
        "chi_bg_model": np.asarray(chi_bg_arr, dtype=np.float32),
        "anomaly_label": np.asarray(model["anomaly_label"], dtype=np.int16),
        "facies_bg": np.asarray(facies_bg, dtype=np.int16) if facies_bg is not None else np.empty((0,), dtype=np.int16),
        "dx": np.array(dx, dtype=np.float32),
        "dy": np.array(dy, dtype=np.float32),
        "dz": np.array(dz, dtype=np.float32),
        "rho_unit": np.array(DENSITY_UNIT),
        "anomaly_type": np.array(model["anomaly_type"]),
        "anomaly_name_en": np.array(model["anomaly_name_en"]),
        "anomaly_name_zh": np.array(model["anomaly_name_zh"]),
        "mt_status": np.array("not_run"),
        "gravity_status": np.array("not_run"),
        "magnetic_status": np.array("not_run"),
        "seismic_status": np.array("not_run"),
        "seismic_preset": np.array(str(seismic_preset)),
    }
    
    # 主模型常驻 CPU，分模块按需搬运，避免某个 CUDA 扩展失败后影响后续模块。
    runtime_device, device_message = _resolve_torch_device(torch_device_preference)
    print(device_message)
    rho_tensor_cpu = torch.tensor(rho_multi, dtype=torch.float32)
    chi_tensor_cpu = torch.tensor(chi_multi, dtype=torch.float32)
    res_tensor_cpu = torch.tensor(res_multi, dtype=torch.float32).contiguous()
    vp_tensor_cpu = torch.tensor(vp_multi, dtype=torch.float32)
    rho_bg_tensor_cpu = torch.tensor(rho_bg_arr, dtype=torch.float32)
    chi_bg_tensor_cpu = torch.tensor(chi_bg_arr, dtype=torch.float32)
    step_idx = 1
    if run_electrical:
        print(f"\n====== {step_idx}. 电磁正演 (MT) ======")
        electrical_preprocess_device = runtime_device
        res_tensor = _move_tensor(res_tensor_cpu, electrical_preprocess_device)
        target_shape = (50, 50, 50)
        print(f"Downsampling MT model from {nx}x{ny}x{nz} to {target_shape[0]}x{target_shape[1]}x{target_shape[2]}...")
        print(f"Original MT model resistivity range: {res_tensor.min().item():.2e} - {res_tensor.max().item():.2e} Ohm-m")
        res_down = _conductivity_domain_downsample(res_tensor, target_shape)
        res_down = _apply_boundary_background_taper(res_down, edge_cells=4)
        
        # 保持物理总尺寸不变，对应放大网格间距
        mt_dx = dx * nx / target_shape[0]
        mt_dy = dy * ny / target_shape[1]
        mt_dz = dz * nz / target_shape[2]
        print(f"New MT spacing: dx={mt_dx:.2f}, dy={mt_dy:.2f}, dz={mt_dz:.2f}")
        print(f"Down sampled res value range : {res_down.min().item():.2e} - {res_down.max().item():.2e} Ohm-m")
        print(f"Down sampled res median / p95 : {torch.median(res_down).item():.2e} / {torch.quantile(res_down, 0.95).item():.2e} Ohm-m")

        _save_mt_xz_slice(
            res_down,
            (mt_dx, mt_dy, mt_dz),
            os.path.join(save_dir, "mt_res_downsampled_xz_slice.png"),
            log_scale=True,
        )
        _save_mt_xz_slice(
            res_down,
            (mt_dx, mt_dy, mt_dz),
            os.path.join(save_dir, "mt_res_downsampled_xz_slice_linear.png"),
            log_scale=False,
        )

        # # 将降采样后的三维电阻率模型保存为 bin 文件，供独立 MTForward3D 项目测试和定位不收敛问题
        # bin_save_path = os.path.join(save_dir, f"downsampled_mt_model_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.bin")
        # # 许多 C++ 反演/正演代码期望的内存顺序为 X 变化最快 (NX内循环)，即在 Python 中看起来像 (NZ, NY, NX)
        # res_down.permute(2, 1, 0).contiguous().cpu().numpy().astype(np.float32).tofile(bin_save_path)
        # print(f"[+] Saved downsampled MT model (bin) to: {bin_save_path}")
        # raise ValueError("MT Forward not implemented yet. This is a placeholder to ensure the pipeline runs up to MT forward. Please implement MT forward or comment out this raise statement to proceed with the rest of the pipeline.")

        mt_solver = ElectricalForwardSolver(None, mt_dx, mt_dy, mt_dz)
        if mt_solver.last_freqs is not None:
            print(f"MT auto-selected {len(mt_solver.last_freqs)} frequencies: {list(mt_solver.last_freqs)}")
        t0 = time.time()
        print(f'res down device type: {res_down.device}, dtype: {res_down.dtype}, shape = {res_down.shape}')
        try:
            # with torch.no_grad():
            app_res, phase = mt_solver(res_down.to(torch.float64))
            print(f"MT completed in {time.time()-t0:.2f}s. AppRes shape: {app_res.shape}, device: {app_res.device}")
            bundle["mt_status"] = np.array("ok")
            bundle["mt_app_res"] = _tensor_to_numpy(app_res).astype(np.float32, copy=False)
            bundle["mt_phase"] = _tensor_to_numpy(phase).astype(np.float32, copy=False)
            bundle["mt_res_model"] = _tensor_to_numpy(res_down).astype(np.float32, copy=False)
            bundle["mt_dx"] = np.array(mt_dx, dtype=np.float32)
            bundle["mt_dy"] = np.array(mt_dy, dtype=np.float32)
            bundle["mt_dz"] = np.array(mt_dz, dtype=np.float32)
            if mt_solver.last_freqs is not None:
                bundle["mt_freqs_hz"] = np.asarray(mt_solver.last_freqs, dtype=np.float32)
        except Exception as e:
            print(f"MT Forward failed (ensure mt_forward_cuda installed): {e}")
            bundle["mt_status"] = np.array("failed")
            bundle["mt_error"] = np.array(str(e))
            if runtime_device.type == "cuda":
                runtime_device = torch.device("cpu")
                print("  --> MT failed after CUDA preprocessing; subsequent Torch modules will continue on CPU.")

    step_idx += 1
    if run_gravity:
        print(f"\n====== {step_idx}. 重力正演 (Gravity) ======")
        gravity_device = runtime_device
        rho_tensor = _move_tensor(rho_tensor_cpu, gravity_device)
        # 控制输入密度以计算相对或绝对重力
        if gravity_anomaly_mode == "background":
            print(f"  --> [Gravity Mode]: 'background'. 计算相对于局部背景模型(岩石骨架)的剩余密度，以获得纯地质体引起的重力异常。")
            rho_bg_tensor = _move_tensor(rho_bg_tensor_cpu, gravity_device)
            gravity_input_density = rho_tensor - rho_bg_tensor
        elif gravity_anomaly_mode == "constant":
            print(f"  --> [Gravity Mode]: 'constant'. 计算相对于常数背景密度 ({gravity_bg_density} {DENSITY_UNIT}) 的密度差，以获得相对重力异常。")
            gravity_input_density = rho_tensor - gravity_bg_density
        else:
            print("  --> [Gravity Mode]: 'absolute'. 输入绝对密度，计算该有限体积网格产生的绝对重力响应 (gz)。")
            gravity_input_density = rho_tensor

        obs_conf_grav = {'layout': 'grid', 'n_x': nx, 'n_y': ny, 'first_x': 0, 'first_y': 0, 'd_x': 1, 'd_y': 1}
        grav_solver = GravityForwardSolver(
            dx, dy, dz, heights_m=[0.0], obs_conf=obs_conf_grav, output_unit="mgal", density_unit=DENSITY_UNIT
        ).to(gravity_device)
        t0 = time.time()
        with torch.no_grad():
            grav_data, _ = grav_solver(gravity_input_density)
        print(f"Gravity completed in {time.time()-t0:.2f}s, min: {grav_data.min():.2f} mGal, max: {grav_data.max():.2f} mGal")
        
        grav_np = grav_data.cpu().numpy().squeeze()
        bundle["gravity_status"] = np.array("ok")
        bundle["gravity_data"] = grav_np.astype(np.float32, copy=False)
        
        plt.figure(figsize=(6,5))
        plt.imshow(grav_np.T, origin='lower', cmap='jet')
        plt.colorbar(label='mGal')
        plt.title('Gravity Anomaly (gz)')
        plt.savefig(os.path.join(save_dir, "forward_gravity.png"), dpi=150)
        plt.close()

    step_idx += 1
    if run_magnetic:
        print(f"\n====== {step_idx}. 磁力正演 (Magnetic) ======")
        magnetic_device = runtime_device
        chi_tensor = _move_tensor(chi_tensor_cpu, magnetic_device)
        # 控制输入磁化率以计算相对或绝对磁力异常
        magnetic_anomaly_mode = gravity_anomaly_mode # 保持与重力相同的扣背景策略(或独立提取配置)
        
        if magnetic_anomaly_mode == "background":
            print(f"  --> [Magnetic Mode]: 'background'. 计算相对于局部背景模型(岩石骨架)的磁化率差，以获得纯地质体的磁异常 (delta T)。")
            chi_bg_tensor = _move_tensor(chi_bg_tensor_cpu, magnetic_device)
            magnetic_input_chi = chi_tensor - chi_bg_tensor
        elif magnetic_anomaly_mode == "constant":
            # 磁力常数背景一般设为 0 (或者某种围岩均值)，此处假设背景均值近似为 0 或微小常数
            constant_chi = 0.0
            print(f"  --> [Magnetic Mode]: 'constant'. 计算相对于常数磁化率 ({constant_chi}) 的相对磁异常。")
            magnetic_input_chi = chi_tensor - constant_chi
        else:
            print("  --> [Magnetic Mode]: 'absolute'. 输入全空间绝对磁化率，计算整体网格产生的总磁场异常。")
            magnetic_input_chi = chi_tensor

        obs_conf_mag = {'layout': 'grid', 'n_x': nx, 'n_y': ny, 'first_x': 0, 'first_y': 0, 'd_x': 1, 'd_y': 1}
        mag_solver = MagneticForwardSolver(dx, dy, dz, heights_m=[0.0], obs_conf=obs_conf_mag, inc=90.0, dec=0.0).to(magnetic_device)
        t0 = time.time()
        with torch.no_grad():
            mag_data, _ = mag_solver(magnetic_input_chi)
        print(f"Magnetic completed in {time.time()-t0:.2f}s, min: {mag_data.min():.2f} nT, max: {mag_data.max():.2f} nT")
        
        mag_np = mag_data.cpu().numpy().squeeze()
        bundle["magnetic_status"] = np.array("ok")
        bundle["magnetic_data"] = mag_np.astype(np.float32, copy=False)
        
        plt.figure(figsize=(6,5))
        plt.imshow(mag_np.T, origin='lower', cmap='jet')
        plt.colorbar(label='nT')
        plt.title('Magnetic Anomaly (TMI)')
        plt.savefig(os.path.join(save_dir, "forward_magnetic.png"), dpi=150)
        plt.close()

    step_idx += 1
    if run_seismic:
        print(f"\n====== {step_idx}. 地震正演 (Seismic - 3D) ======")
        seismic_device = runtime_device
        if seismic_device.type == "cpu":
            print("  --> Seismic device switched to CPU; 3D simulation may be very slow.")
        vp_3d = _move_tensor(vp_tensor_cpu, seismic_device)
        seismic_config = _build_seismic_3d_config((nx, ny, nz), (dx, dy, dz), preset=seismic_preset)
        source_locations, receiver_locations = setup_acquisition(seismic_config, seismic_device)
        wavelet = get_wavelet(seismic_config, seismic_device)
        dt = float(seismic_config["time"]["dt"])
        nt = int(seismic_config["time"]["nt"])
        freq_hz = float(seismic_config["source"]["freq"])
        n_shots = int(source_locations.shape[0])
        n_src = int(source_locations.shape[1])
        n_rec = int(receiver_locations.shape[1])
        print(
            f"  --> Seismic preset: {seismic_preset} | shots={n_shots} "
            f"({seismic_config['source']['n_shots_x']}x{seismic_config['source']['n_shots_y']}), "
            f"receivers/shot={n_rec} "
            f"({seismic_config['receiver']['n_receivers_x']}x{seismic_config['receiver']['n_receivers_y']})"
        )
        src_amp_single = wavelet.view(1, 1, nt).repeat(1, n_src, 1)
        bundle["seismic_source_locations"] = _tensor_to_numpy(source_locations).astype(np.int16, copy=False)
        bundle["seismic_receiver_locations"] = _tensor_to_numpy(receiver_locations).astype(np.int16, copy=False)
        bundle["seismic_wavelet"] = _tensor_to_numpy(wavelet).astype(np.float32, copy=False)
        bundle["seismic_dt"] = np.array(dt, dtype=np.float32)
        bundle["seismic_nt"] = np.array(nt, dtype=np.int32)
        bundle["seismic_freq_hz"] = np.array(freq_hz, dtype=np.float32)
        bundle["seismic_mode"] = np.array("3D")

        t0 = time.time()
        try:
            shot_gathers = []
            for shot_idx in range(n_shots):
                if shot_idx == 0 or (shot_idx + 1) % 5 == 0 or shot_idx + 1 == n_shots:
                    print(f"  --> Seismic shot {shot_idx + 1}/{n_shots}")
                seis_solver = SeismicForwardSolver(
                    [dx, dy, dz],
                    dt,
                    src_amp_single,
                    source_locations[shot_idx:shot_idx + 1],
                    receiver_locations[shot_idx:shot_idx + 1],
                    pml_freq=freq_hz,
                    accuracy=int(seismic_config["simulation"].get("accuracy", 4)),
                ).to(seismic_device)
                with torch.no_grad():
                    out = seis_solver(vp_3d)
                    shot_gathers.append(out[-1].detach().cpu())
                if seismic_device.type == "cuda":
                    torch.cuda.empty_cache()

            seis_data = torch.cat(shot_gathers, dim=0)
            print(f"Seismic 3D completed in {time.time()-t0:.2f}s. Data shape: {seis_data.shape}")
            
            seis_np = seis_data.cpu().numpy().astype(np.float32, copy=False)
            bundle["seismic_status"] = np.array("ok")
            bundle["seismic_data"] = seis_np
            vmin, vmax = np.percentile(seis_np, [0.5, 99.5])
            
            plt.figure(figsize=(6,5))
            plt.imshow(seis_np[0].T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
            plt.title('Seismic Shot Gather (3D, Shot 0)')
            plt.savefig(os.path.join(save_dir, "forward_seismic.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Seismic Forward failed (ensure Deepwave installed): {e}")
            bundle["seismic_status"] = np.array("failed")
            bundle["seismic_error"] = np.array(str(e))
            
    bundle_path = os.path.join(save_dir, "forward_bundle.npz")
    np.savez(bundle_path, **bundle)
    print(f"[+] Saved unified forward bundle to: {bundle_path}")
    print("\n====== All Done! ======")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./DATAFOLDER/Cache/ForwardOutput")
    parser.add_argument("--vp_segy", type=str, default="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy", help="Path to background Vp SEGY.")
    parser.add_argument("--label_segy", type=str, default="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy", help="Path to background label SEGY.")
    parser.add_argument("--skip_seismic", action="store_true")
    parser.add_argument("--skip_mt", action="store_true")
    parser.add_argument("--anomaly-type", dest="anomaly_type", type=str, default="igneous_swarm", choices=FORWARD_ANOMALY_TYPES, help="Anomaly to inject for forward modeling.")
    parser.add_argument("--seismic-preset", dest="seismic_preset", type=str, default="full", choices=SEISMIC_PRESETS, help="3D seismic acquisition size preset.")
    parser.add_argument("--anomaly_mode", dest="gravity_anomaly_mode", type=str, default="background", choices=["absolute", "background", "constant"], help="Forward mode for input density/susceptibility: absolute, background, or constant.")
    parser.add_argument("--bg_density", dest="gravity_bg_density", type=float, default=2.67, help="Constant bulk density in g/cm^3 to subtract for 'constant' mode.")
    parser.add_argument("--device", dest="torch_device_preference", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for Torch-based modules (gravity/magnetic/seismic). MT keeps its own CUDA backend.")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    run_forward_pipeline(
        args.save_dir, 
        args.vp_segy,
        args.label_segy,
        run_electrical=not args.skip_mt, 
        run_seismic=not args.skip_seismic,
        gravity_anomaly_mode=args.gravity_anomaly_mode,
        gravity_bg_density=args.gravity_bg_density,
        torch_device_preference=args.torch_device_preference,
        anomaly_type=args.anomaly_type,
        seismic_preset=args.seismic_preset,
    )
