import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cigvis

def plot_2d_outputs(save_dir):
    """
    加载并绘制重力、磁力、地震的 2D 剖面结果
    """
    print("\n--- 绘制物理场正演剖面 ---")
    fig_dir = os.path.join(save_dir, "custom_plots")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. 重力
    grav_path = os.path.join(save_dir, "forward_gravity.npy")
    if os.path.exists(grav_path):
        grav = np.load(grav_path)
        plt.figure(figsize=(6, 5))
        plt.imshow(grav.T, origin='lower', cmap='jet')
        plt.colorbar(label='mGal')
        plt.title('Gravity Anomaly')
        out_path = os.path.join(fig_dir, 'custom_gravity.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[+] Reloaded Gravity -> {out_path}")
        
    # 2. 磁力
    mag_path = os.path.join(save_dir, "forward_magnetic.npy")
    if os.path.exists(mag_path):
        mag = np.load(mag_path)
        plt.figure(figsize=(6, 5))
        plt.imshow(mag.T, origin='lower', cmap='jet')
        plt.colorbar(label='nT')
        plt.title('Magnetic Anomaly')
        out_path = os.path.join(fig_dir, 'custom_magnetic.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[+] Reloaded Magnetic -> {out_path}")

    # 3. 地震
    seis_path = os.path.join(save_dir, "forward_seismic.npy")
    if os.path.exists(seis_path):
        seis = np.load(seis_path)
        plt.figure(figsize=(6, 5))
        plt.imshow(seis[0].T, aspect='auto', cmap='gray', vmin=-1, vmax=1)
        plt.title('Seismic Shot Gather')
        plt.xlabel('Receiver Index')
        plt.ylabel('Time Step')
        out_path = os.path.join(fig_dir, 'custom_seismic.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[+] Reloaded Seismic -> {out_path}")

def plot_3d_mt_with_cigvis(save_dir, run_app=False):
    """
    使用 cigvis 将正演 MT 结果(各频率的视电阻率与相位、TE与TM极化模式)在三维视图中展开
    视电阻率使用 log10 显示以突出量级变化。
    由于正演输出为 [N_freq, nx, ny, 2]，我们需要为不同频率将其三维化/切片化展示
    """
    mt_app_path = os.path.join(save_dir, "forward_mt_app_res.npy")
    mt_pha_path = os.path.join(save_dir, "forward_mt_phase.npy")
    if not (os.path.exists(mt_app_path) and os.path.exists(mt_pha_path)):
        print(f"[-] MT 相关数据不存在，跳过 MT 3D 渲染。")
        return
        
    print("\n--- 绘制 MT 全空间张量 (Cigvis 3D) ---")
    app_res = np.load(mt_app_path) # [N_freq, nx, ny, 2(TE/TM)]
    phase = np.load(mt_pha_path)   # [N_freq, nx, ny, 2(TE/TM)]
    
    N_freq, nx, ny, _ = app_res.shape
    
    # 我们将把频率作为假的 Z 轴（深度向）来叠加显示三维频率切片。
    # 重组为 [nx, ny, N_freq]
    # 取 log 后视电阻率
    log_app_res = np.log10(np.clip(app_res, 1e-6, None))
    
    # 抽取 TE 模式 (index 0) 和 TM 模式 (index 1)
    TE_app_vol = log_app_res[..., 0].transpose(1, 2, 0) # [nx, ny, N_freq]
    TM_app_vol = log_app_res[..., 1].transpose(1, 2, 0)
    
    TE_pha_vol = phase[..., 0].transpose(1, 2, 0)
    TM_pha_vol = phase[..., 1].transpose(1, 2, 0)
    
    def _safe_clim(vmin, vmax, eps=1e-5):
        return [vmin, vmax] if vmax > vmin else [vmin, vmax + eps]

    n_te_app = cigvis.create_slices(TE_app_vol, cmap='jet', clim=_safe_clim(TE_app_vol.min(), TE_app_vol.max()))
    n_tm_app = cigvis.create_slices(TM_app_vol, cmap='jet', clim=_safe_clim(TM_app_vol.min(), TM_app_vol.max()))
    n_te_pha = cigvis.create_slices(TE_pha_vol, cmap='jet', clim=_safe_clim(TE_pha_vol.min(), TE_pha_vol.max()))
    n_tm_pha = cigvis.create_slices(TM_pha_vol, cmap='jet', clim=_safe_clim(TM_pha_vol.min(), TM_pha_vol.max()))

    # 为其增加 Colorbar
    cb_te_app = cigvis.create_colorbar_from_nodes(n_te_app, label_str="TE AppRes (log10)")
    cb_tm_app = cigvis.create_colorbar_from_nodes(n_tm_app, label_str="TM AppRes (log10)")
    cb_te_pha = cigvis.create_colorbar_from_nodes(n_te_pha, label_str="TE Phase (deg)")
    cb_tm_pha = cigvis.create_colorbar_from_nodes(n_tm_pha, label_str="TM Phase (deg)")
    
    for nds, cb in zip((n_te_app, n_tm_app, n_te_pha, n_tm_pha), (cb_te_app, cb_tm_app, cb_te_pha, cb_tm_pha)):
        if isinstance(cb, list): nds.extend(cb)
        else: nds.append(cb)

    fig_dir = os.path.join(save_dir, "custom_plots")
    out_file_mt = "Replot_3D_MT.png"
    
    cigvis.plot3D(
        [n_te_app, n_tm_app, n_te_pha, n_tm_pha], 
        grid=(2, 2), 
        savename=out_file_mt, 
        savedir=fig_dir, 
        run_app=run_app,
        title=["TE AppRes (log10) Freq-Slice", "TM AppRes (log10) Freq-Slice", "TE Phase", "TM Phase"]
    )
    print(f"[+] Reloaded 3D MT Tensors -> {os.path.join(fig_dir, out_file_mt)}")


def plot_3d_models(save_dir, run_app=False):
    """
    加载模型基础数据 (.npz)，并使用 cigvis 二次将其在3D空间中渲染
    """
    model_path = os.path.join(save_dir, "forward_models.npz")
    if not os.path.exists(model_path):
        print(f"[-] {model_path} 不存在，跳过 3D 渲染。")
        return
        
    print("\n--- 绘制模型 3D 空间结构 ---")
    data = np.load(model_path)
    vp = data['vp']
    rho = data['rho']
    res = data['res']
    chi = data['chi'] * 1e5  # 放大系数便于成图，与原逻辑一致
    
    def _safe_clim(vmin, vmax, eps=1e-5):
        return [vmin, vmax] if vmax > vmin else [vmin, vmax + eps]

    log_res = np.log10(np.clip(res, a_min=1e-5, a_max=None))
    
    nodes_vp = cigvis.create_slices(vp, cmap='jet', clim=_safe_clim(vp.min(), vp.max()))
    nodes_rho = cigvis.create_slices(rho, cmap='jet', clim=_safe_clim(rho.min(), rho.max()))
    nodes_res = cigvis.create_slices(log_res, cmap='jet', clim=_safe_clim(log_res.min(), log_res.max()))
    nodes_chi = cigvis.create_slices(chi, cmap='jet', clim=_safe_clim(chi.min(), chi.max()))

    fig_dir = os.path.join(save_dir, "custom_plots")
    os.makedirs(fig_dir, exist_ok=True)
    out_file_3d = "Replot_3D_Models.png"

    cigvis.plot3D(
        [nodes_vp, nodes_rho, nodes_res, nodes_chi], 
        grid=(2, 2), 
        savename=out_file_3d, 
        savedir=fig_dir, 
        run_app=run_app, 
        title=["Loaded Vp (m/s)", "Loaded Density (kg/m^3)", "Loaded log10(Resistivity)", "Loaded Susceptibility (x1e-5)"]
    )
    print(f"[+] Reloaded 3D Models -> {os.path.join(fig_dir, out_file_3d)}")


def get_mt_freqs():
    import numpy as np
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
    return np.array(freqs)

def plot_2d_mt_outputs(save_dir):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    print("\n--- 绘制 MT 2D 视电阻率与相位剖面和切片 ---")
    fig_dir = os.path.join(save_dir, "custom_plots")
    os.makedirs(fig_dir, exist_ok=True)
    
    mt_app_path = os.path.join(save_dir, "forward_mt_app_res.npy")
    mt_pha_path = os.path.join(save_dir, "forward_mt_phase.npy")
    model_path = os.path.join(save_dir, "forward_models.npz")
    
    if not (os.path.exists(mt_app_path) and os.path.exists(mt_pha_path)):
        print(f"[-] MT 相关数据不存在，跳过 MT 2D 绘图。")
        return
        
    app_res = np.load(mt_app_path) # [N_freq, nx, ny, 2(TE/TM)]
    phase = np.load(mt_pha_path)   # [N_freq, nx, ny, 2(TE/TM)]
    N_freq, nx, ny, _ = app_res.shape
    freqs = get_mt_freqs()
    if len(freqs) != N_freq:
        freqs = np.arange(N_freq)
        
    dx, dy = 160.0, 160.0
    if os.path.exists(model_path):
        data = np.load(model_path)
        orig_nx = data['vp'].shape[0]
        orig_dx = data['dx']
        orig_dy = data['dy']
        dx = orig_dx * orig_nx / nx
        dy = orig_dy * data['vp'].shape[1] / ny
        
    x_1d = (np.arange(nx) + 0.5) * dx
    y_1d = (np.arange(ny) + 0.5) * dy
    X_grid, Y_grid = np.meshgrid(x_1d, y_1d, indexing='ij')

    phi_xy_plot = np.abs(phase[..., 0])
    phi_yx_plot = phase[..., 1] + 180.0
    
    cx, cy = nx // 2, ny // 2
    f_station = freqs
    rxy_station = app_res[:, cx, cy, 0]
    ryx_station = app_res[:, cx, cy, 1]
    pxy_station = phi_xy_plot[:, cx, cy]
    pyx_station = phi_yx_plot[:, cx, cy]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    ax1.loglog(f_station, rxy_station, 'o-', color='red', label=r'$\rho_{xy}$ (TE)')
    ax1.loglog(f_station, ryx_station, 's-', color='blue', label=r'$\rho_{yx}$ (TM)')
    ax1.set_ylabel(r'Apparent Resistivity ($\Omega \cdot m$)')
    ax1.set_title(f'MT Sounding Curves at Center Station (X={x_1d[cx]:.1f}m, Y={y_1d[cy]:.1f}m)')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    if freqs[0] > freqs[-1]:
        ax1.invert_xaxis()
        
    ax2.semilogx(f_station, pxy_station, 'o-', color='red', label=r'$\phi_{xy}$ (TE)')
    ax2.semilogx(f_station, pyx_station, 's-', color='blue', label=r'$\phi_{yx}$ (TM)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (Degrees)')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    
    plt.tight_layout()
    out_curve = os.path.join(fig_dir, "custom_mt_sounding.png")
    plt.savefig(out_curve, dpi=150)
    plt.close()
    print(f"[+] Reloaded MT Sounding Curves -> {out_curve}")
    
    mid_f_idx = N_freq // 2
    freq_to_plot = freqs[mid_f_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    im1 = ax1.pcolormesh(X_grid, Y_grid, np.log10(app_res[mid_f_idx, :, :, 0]), shading='nearest', cmap='jet')
    fig.colorbar(im1, ax=ax1, label=r'Log10($\rho$) ($\Omega \cdot m$)')
    ax1.set_title(rf'$\log_{{10}} \rho_{{xy}}$ (TE) at {freq_to_plot:.3f} Hz')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    
    im2 = ax2.pcolormesh(X_grid, Y_grid, np.log10(app_res[mid_f_idx, :, :, 1]), shading='nearest', cmap='jet')
    fig.colorbar(im2, ax=ax2, label=r'Log10($\rho$) ($\Omega \cdot m$)')
    ax2.set_title(rf'$\log_{{10}} \rho_{{yx}}$ (TM) at {freq_to_plot:.3f} Hz')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    
    plt.tight_layout()
    out_map = os.path.join(fig_dir, "custom_mt_map_2d.png")
    plt.savefig(out_map, dpi=150)
    plt.close()
    print(f"[+] Reloaded MT 2D Map -> {out_map}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    X_mesh, F_mesh = np.meshgrid(x_1d, freqs) # shape (nx, N_freq)
    # X_mesh = X_mesh.T
    # F_mesh = F_mesh.T
    
    rho_xy_sec = app_res[:, :, cy, 0] 
    rho_yx_sec = app_res[:, :, cy, 1]
    phi_xy_sec = phi_xy_plot[:, :, cy]
    phi_yx_sec = phi_yx_plot[:, :, cy]
    
    im1 = axes[0, 0].pcolormesh(X_mesh, F_mesh, np.log10(rho_xy_sec), shading='nearest', cmap='jet')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_title(rf'$\log_{{10}} \rho_{{xy}}$ (TE) at Y = {y_1d[cy]} m')
    fig.colorbar(im1, ax=axes[0, 0], label=r'Log10($\rho$) ($\Omega \cdot m$)')
    if freqs[0] > freqs[-1]:
        axes[0,0].invert_yaxis()

    im2 = axes[0, 1].pcolormesh(X_mesh, F_mesh, np.log10(rho_yx_sec), shading='nearest', cmap='jet')
    axes[0, 1].set_title(rf'$\log_{{10}} \rho_{{yx}}$ (TM) at Y = {y_1d[cy]} m')
    fig.colorbar(im2, ax=axes[0, 1], label=r'Log10($\rho$) ($\Omega \cdot m$)')

    im3 = axes[1, 0].pcolormesh(X_mesh, F_mesh, phi_xy_sec, shading='nearest', cmap='jet')
    axes[1, 0].set_xlabel('X Coordinate (m)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_title(rf'Phase $\phi_{{xy}}$ (TE) at Y = {y_1d[cy]} m')
    fig.colorbar(im3, ax=axes[1, 0], label='Phase (Degrees)')

    im4 = axes[1, 1].pcolormesh(X_mesh, F_mesh, phi_yx_sec, shading='nearest', cmap='jet')
    axes[1, 1].set_xlabel('X Coordinate (m)')
    axes[1, 1].set_title(rf'Phase $\phi_{{yx}}$ (TM) at Y = {y_1d[cy]} m')
    fig.colorbar(im4, ax=axes[1, 1], label='Phase (Degrees)')

    plt.tight_layout()
    out_sec = os.path.join(fig_dir, "custom_mt_pseudosection.png")
    plt.savefig(out_sec, dpi=150)
    plt.close()
    print(f"[+] Reloaded MT Pseudosection -> {out_sec}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Load and plot saved NumPy files from Multi-physics forward pipeline.")
    parser.add_argument("--save_dir", type=str, default="./DATAFOLDER/Cache/ForwardOutput", help="Directory where .npy and .npz outputs are located.")
    parser.add_argument("--run_app", action="store_true", help="Launch interactive 3D GUI for the re-plotted models.")
    parser.add_argument("--skip_mt", action="store_true", help="Skip MT 2D plotting.")
    args = parser.parse_args()
    
    plot_2d_outputs(args.save_dir)
    if not args.skip_mt:
        plot_2d_mt_outputs(args.save_dir)
    plot_3d_mt_with_cigvis(args.save_dir, run_app=args.run_app)
    plot_3d_models(args.save_dir, run_app=args.run_app)
