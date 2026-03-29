import argparse
import os
import time
import numpy as np
import torch
import segyio
import matplotlib.pyplot as plt

from core.builder import DatasetBuilder
from core.petrophysics.rock_physics import PetrophysicsConverter
from core.anomalies.hydrocarbon_hydrate import HydrocarbonHydrate, HydrocarbonHydrateParams

from core.forward_modeling.gravity import GravityForwardSolver
from core.forward_modeling.magnetic import MagneticForwardSolver
from core.forward_modeling.electrical import ElectricalForwardSolver
from core.forward_modeling.seismic import SeismicForwardSolver


DENSITY_UNIT = "kg/m^3"


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

def read_segy_volume(path):
    print(f"Reading SEGY: {path}")
    with segyio.open(path, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as f:
        try:
            vol = segyio.tools.cube(f)
        except Exception:
            f.mmap()
            vol = segyio.tools.cube(f)
    dx, dy, dz = 10.0, 10.0, 25.0
    return vol, (dx, dy, dz)

def generate_model(vp_segy_path, label_segy_path):
    """ 读取 SEGY 构建全尺寸带块状硫化物的四参数模型用于打通正演管线 """
    try:
        vp_bg, (dx, dy, dz) = read_segy_volume(vp_segy_path)
        label_vol, _ = read_segy_volume(label_segy_path)
        nx, ny, nz = vp_bg.shape
        print(f'Velocity shape = {nx, ny, nz}, dx={dx}, dy={dy}, dz={dz}')
    except Exception as e:
        raise RuntimeError(f"Failed to read SEGY volumes: {e}")
    
    # 使用 viz 脚本中的气藏 (Gas Reservoir) 异常体参数
    gas_params = HydrocarbonHydrateParams(
        kind="gas", layer_id=-1, center_x_m=1500.0, center_y_m=1500.0,
        lens_extent_x_m=1200.0, lens_extent_y_m=700.0, lens_thickness_m=120.0, vp_gas_mps=1800.0,
        gas_enable_chimney=True, chimney_height_m=1200.0, rng_seed=11
    )
    anomaly = HydrocarbonHydrate(params=gas_params, layer_labels=label_vol)
    
    converter = PetrophysicsConverter()
    rho_bg, res_bg, chi_bg = converter.generate_background(vp_bg, label_vol=label_vol)
    background_state = converter.get_last_background_state()
    print(f'背景密度 value range = {rho_bg.min():.2f} - {rho_bg.max():.2f} {DENSITY_UNIT}')
    print(f'背景电阻率 value range = {res_bg.min():.2e} - {res_bg.max():.2e} Ohm-m')
    
    builder = DatasetBuilder(dx, dy, dz)
    _, mask_final, _, _, _ = builder.inject_anomalies(vp_bg.copy(), [anomaly])
    mask_bool = mask_final > 0
    
    vp_multi, rho_multi, res_multi, chi_multi = converter.apply_anomaly(
        mask_bool, "Gas", vp_bg.copy(), rho_bg.copy(), res_bg.copy(), chi_bg.copy()
    )
    updated_state = converter.get_last_background_state()
    return (
        vp_multi,
        rho_multi,
        res_multi,
        chi_multi,
        rho_bg,
        chi_bg,
        (nx, ny, nz),
        (dx, dy, dz),
        background_state["facies"].astype(np.int16) if background_state is not None else None,
        updated_state["facies"].astype(np.int16) if updated_state is not None else None,
    )

def run_forward_pipeline(save_dir, vp_segy_path, label_segy_path, run_gravity=True, run_magnetic=True, run_electrical=True, run_seismic=True, gravity_anomaly_mode="background", gravity_bg_density=2670.0):
    print("====== 1. 模型生成 ======")
    vp_multi, rho_multi, res_multi, chi_multi, rho_bg_arr, chi_bg_arr, shape, spacing, facies_bg, facies_multi = generate_model(vp_segy_path, label_segy_path)
    nx, ny, nz = shape
    dx, dy, dz = spacing
    print(f"Generated Models: {nx}x{ny}x{nz} (dx={dx}, dy={dy}, dz={dz})")
    
    # 保存生成的物性模型本身供独立制图或检验
    np.savez(os.path.join(save_dir, "forward_models.npz"), 
             vp=vp_multi, rho=rho_multi, res=res_multi, chi=chi_multi,
             facies_bg=facies_bg, facies=facies_multi,
             dx=dx, dy=dy, dz=dz, rho_unit=DENSITY_UNIT)
    
    # 转为 PyTorch Tensor 并放到 GPU (如支持)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rho_tensor = torch.tensor(rho_multi, dtype=torch.float32, device=device)
    chi_tensor = torch.tensor(chi_multi, dtype=torch.float32, device=device)
    res_tensor = torch.tensor(res_multi, dtype=torch.float32, device=device).contiguous()
    vp_tensor  = torch.tensor(vp_multi, dtype=torch.float32, device=device)
    
    if run_gravity:
        print("\n====== 2. 重力正演 (Gravity) ======")
        # 控制输入密度以计算相对或绝对重力
        if gravity_anomaly_mode == "background":
            print(f"  --> [Gravity Mode]: 'background'. 计算相对于局部背景模型(岩石骨架)的剩余密度，以获得纯地质体引起的重力异常。")
            rho_bg_tensor = torch.tensor(rho_bg_arr, dtype=torch.float32, device=device)
            gravity_input_density = rho_tensor - rho_bg_tensor
        elif gravity_anomaly_mode == "constant":
            print(f"  --> [Gravity Mode]: 'constant'. 计算相对于常数背景密度 ({gravity_bg_density} kg/m^3) 的密度差，以获得相对重力异常。")
            gravity_input_density = rho_tensor - gravity_bg_density
        else:
            print("  --> [Gravity Mode]: 'absolute'. 输入绝对密度，计算该有限体积网格产生的绝对重力响应 (gz)。")
            gravity_input_density = rho_tensor

        obs_conf_grav = {'layout': 'grid', 'n_x': nx, 'n_y': ny, 'first_x': 0, 'first_y': 0, 'd_x': 1, 'd_y': 1}
        grav_solver = GravityForwardSolver(dx, dy, dz, heights_m=[0.0], obs_conf=obs_conf_grav, output_unit="mgal").to(device)
        t0 = time.time()
        with torch.no_grad():
            grav_data, _ = grav_solver(gravity_input_density)
        print(f"Gravity completed in {time.time()-t0:.2f}s, min: {grav_data.min():.2f} mGal, max: {grav_data.max():.2f} mGal")
        
        grav_np = grav_data.cpu().numpy().squeeze()
        np.save(os.path.join(save_dir, "forward_gravity.npy"), grav_np)
        
        plt.figure(figsize=(6,5))
        plt.imshow(grav_np.T, origin='lower', cmap='jet')
        plt.colorbar(label='mGal')
        plt.title('Gravity Anomaly (gz)')
        plt.savefig(os.path.join(save_dir, "forward_gravity.png"), dpi=150)
        plt.close()

    if run_magnetic:
        print("\n====== 3. 磁力正演 (Magnetic) ======")
        # 控制输入磁化率以计算相对或绝对磁力异常
        magnetic_anomaly_mode = gravity_anomaly_mode # 保持与重力相同的扣背景策略(或独立提取配置)
        
        if magnetic_anomaly_mode == "background":
            print(f"  --> [Magnetic Mode]: 'background'. 计算相对于局部背景模型(岩石骨架)的磁化率差，以获得纯地质体的磁异常 (delta T)。")
            chi_bg_tensor = torch.tensor(chi_bg_arr, dtype=torch.float32, device=device)
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
        mag_solver = MagneticForwardSolver(dx, dy, dz, heights_m=[0.0], obs_conf=obs_conf_mag, inc=90.0, dec=0.0).to(device)
        t0 = time.time()
        with torch.no_grad():
            mag_data, _ = mag_solver(magnetic_input_chi)
        print(f"Magnetic completed in {time.time()-t0:.2f}s, min: {mag_data.min():.2f} nT, max: {mag_data.max():.2f} nT")
        
        mag_np = mag_data.cpu().numpy().squeeze()
        np.save(os.path.join(save_dir, "forward_magnetic.npy"), mag_np)
        
        plt.figure(figsize=(6,5))
        plt.imshow(mag_np.T, origin='lower', cmap='jet')
        plt.colorbar(label='nT')
        plt.title('Magnetic Anomaly (TMI)')
        plt.savefig(os.path.join(save_dir, "forward_magnetic.png"), dpi=150)
        plt.close()

    if run_electrical:
        print("\n====== 4. 电磁正演 (MT) ======")
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

        # 可视化 x-z 剖面 (y轴中心) 检查电阻率模型是否合理
        plt.figure(figsize=(8, 5))
        # res_down 形状为 (nx, ny, nz)
        mid_y = target_shape[1] // 2
        slice_xz = res_down[:, mid_y, :].cpu().numpy()
        # 为了符合常规的地质剖面视角 (Z轴向下)，将其转置并设置 extent
        plt.imshow(np.log10(slice_xz.T), aspect='auto', cmap='jet', 
                   extent=[0, target_shape[0]*mt_dx, target_shape[2]*mt_dz, 0])
        plt.colorbar(label='Log10 Resistivity (Ohm-m)')
        plt.title(f'Downsampled MT Resistivity X-Z Slice (y={mid_y})')
        plt.xlabel('X (m)')
        plt.ylabel('Depth (m)')
        slice_save_path = os.path.join(save_dir, "mt_res_downsampled_xz_slice.png")
        plt.savefig(slice_save_path, dpi=150)
        plt.close()
        print(f"[+] Saved MT resistivity X-Z slice to: {slice_save_path}")
        
        # 可视化 x-z 剖面 (y轴中心) 检查电阻率模型是否合理
        plt.figure(figsize=(8, 5))
        # res_down 形状为 (nx, ny, nz)
        mid_y = target_shape[1] // 2
        slice_xz = res_down[:, mid_y, :].cpu().numpy()
        # 为了符合常规的地质剖面视角 (Z轴向下)，将其转置并设置 extent
        plt.imshow(slice_xz.T, aspect='auto', cmap='jet', 
                   extent=[0, target_shape[0]*mt_dx, target_shape[2]*mt_dz, 0])
        plt.colorbar(label='Resistivity (Ohm-m)')
        plt.title(f'Downsampled MT Resistivity X-Z Slice (y={mid_y})')
        plt.xlabel('X (m)')
        plt.ylabel('Depth (m)')
        slice_save_path = os.path.join(save_dir, "mt_res_downsampled_xz_slice——nolog.png")
        plt.savefig(slice_save_path, dpi=150)
        plt.close()
        print(f"[+] Saved MT resistivity X-Z slice to: {slice_save_path}")

        # 将降采样后的三维电阻率模型保存为 bin 文件，供独立 MTForward3D 项目测试和定位不收敛问题
        bin_save_path = os.path.join(save_dir, f"downsampled_mt_model_{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.bin")
        # 许多 C++ 反演/正演代码期望的内存顺序为 X 变化最快 (NX内循环)，即在 Python 中看起来像 (NZ, NY, NX)
        res_down.permute(2, 1, 0).contiguous().cpu().numpy().astype(np.float32).tofile(bin_save_path)
        print(f"[+] Saved downsampled MT model (bin) to: {bin_save_path}")
        raise ValueError("MT Forward not implemented yet. This is a placeholder to ensure the pipeline runs up to MT forward. Please implement MT forward or comment out this raise statement to proceed with the rest of the pipeline.")

        f_min, f_max = 0.01, 1000.0
        phoenix_coeffs = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]
        max_power = int(np.ceil(np.log10(f_max)))
        min_power = int(np.floor(np.log10(f_min)))
        freqs = []
        for p in range(max_power, min_power - 1, -1):
            power_of_10 = 10.0 ** p
            for coeff in phoenix_coeffs:
                current_f = float(coeff * power_of_10)
                if current_f <= f_max * 1.0001 and current_f >= f_min * 0.9999:
                    freqs.append(current_f)
        mt_solver = ElectricalForwardSolver(freqs, mt_dx, mt_dy, mt_dz).to(device)
        t0 = time.time()
        try:
            with torch.no_grad():
                app_res, phase = mt_solver(res_down.to(torch.float64))
            print(f"MT completed in {time.time()-t0:.2f}s. AppRes shape: {app_res.shape}")
            
            # Save MT outputs
            np.save(os.path.join(save_dir, "forward_mt_app_res.npy"), app_res.cpu().numpy())
            np.save(os.path.join(save_dir, "forward_mt_phase.npy"), phase.cpu().numpy())
        except Exception as e:
            print(f"MT Forward failed (ensure mt_forward_cuda installed): {e}")

    if run_seismic:
        print("\n====== 5. 地震正演 (Seismic - 2D Slice Demo) ======")
        # 3D Deepwave costs huge memory, taking a 2D slice for demo.
        vp_2d = vp_tensor[:, int(ny/2), :] # [nx, nz]
        dt = 0.001
        nt = 1000
        # Source at center surface
        src_loc = torch.tensor([[[int(nx/2), 0]]], dtype=torch.long, device=device) # [1, 1, 2]
        # Receivers along surface
        rec_loc = torch.zeros((1, nx, 2), dtype=torch.long, device=device)
        rec_loc[0, :, 0] = torch.arange(nx)
        
        src_amp = (torch.sin(torch.arange(nt, dtype=torch.float32)*0.1) * 
                  torch.exp(-((torch.arange(nt, dtype=torch.float32)-50)/20)**2)).view(1, 1, nt).to(device)
                  
        seis_solver = SeismicForwardSolver(dx, dt, src_amp, src_loc, rec_loc, pml_width=10).to(device)
        t0 = time.time()
        try:
            with torch.no_grad():
                out = seis_solver(vp_2d)
                # Deepwave returns a list: [..., receiver_amplitudes]
                seis_data = out[-1]
            print(f"Seismic 2D completed in {time.time()-t0:.2f}s. Data shape: {seis_data.shape}")
            
            seis_np = seis_data.cpu().numpy()
            np.save(os.path.join(save_dir, "forward_seismic.npy"), seis_np)
            
            plt.figure(figsize=(6,5))
            # [1, N_rec, Nt]
            plt.imshow(seis_np[0].T, aspect='auto', cmap='gray', vmin=-1, vmax=1)
            plt.title('Seismic Shot Gather (2D Slice)')
            plt.savefig(os.path.join(save_dir, "forward_seismic.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Seismic Forward failed (ensure Deepwave installed): {e}")
            
    print("\n====== All Done! ======")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./DATAFOLDER/Cache/ForwardOutput")
    parser.add_argument("--vp_segy", type=str, default="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy", help="Path to background Vp SEGY.")
    parser.add_argument("--label_segy", type=str, default="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy", help="Path to background label SEGY.")
    parser.add_argument("--skip_seismic", action="store_true")
    parser.add_argument("--skip_mt", action="store_true")
    parser.add_argument("--anomaly_mode", dest="gravity_anomaly_mode", type=str, default="background", choices=["absolute", "background", "constant"], help="Forward mode for input density/susceptibility: absolute, background, or constant.")
    parser.add_argument("--bg_density", dest="gravity_bg_density", type=float, default=2670.0, help="Constant bulk density to subtract for 'constant' mode.")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    run_forward_pipeline(
        args.save_dir, 
        args.vp_segy,
        args.label_segy,
        run_electrical=not args.skip_mt, 
        run_seismic=not args.skip_seismic,
        gravity_anomaly_mode=args.gravity_anomaly_mode,
        gravity_bg_density=args.gravity_bg_density
    )
