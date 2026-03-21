import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from mt_forward import MTForward3D

if __name__ == "__main__":
    filepath = '/home/wangyh/Project/GMESUni/MTForward3D/MTModel/em_model_3d_50x30x50.bin'
    
    NX, NY, NZ = 50, 30, 50
    dx, dy, dz = 160.0, 160.0, 80.0
    f_min, f_max = 0.01, 1000.0

    print(f"Loading binary model from {filepath}")
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    # In the binary file, X changes fastest, then Y, then Z. (from C++ elemIdx = i + j*NX + k*NX*NY)
    # Numpy's shape (NZ, NY, NX) correctly maps to this memory layout.
    rho_tensor = torch.from_numpy(data.astype(np.float64)).view(NZ, NY, NX).permute(2, 1, 0).contiguous()

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
                
    if not freqs:
        freqs = [f_max, f_min]
        
    print(f"Calculated {len(freqs)} Frequencies to solve.")
    operator = MTForward3D(freqs, dx, dy, dz)
    
    # ==========================================
    # Run MT 3D Forward (Single GPU Original Version)
    # ==========================================
    print("\n====================================")
    print(f"🚀 Running Single-GPU Execution")
    print("====================================")
    start_multi = time.time()
    
    app_res_multi, phase_multi = operator(rho_tensor)
    end_multi = time.time()
    time_multi = end_multi - start_multi
    print(f"✅ Executed 3D MT Forward in: {time_multi:.2f} s")
    
    # ==========================================
    # Visualization
    # ==========================================
    print("\n====================================")
    print("📊 Generating Visualizations...")
    print("====================================")
    
    app_res = app_res_multi.numpy()    # (n_freqs, NY, NX, 2)
    phase = phase_multi.numpy()        # (n_freqs, NY, NX, 2)
    freqs_np = np.array(freqs)
    
    x_coords = (np.arange(NX) + 0.5) * dx
    y_coords = (np.arange(NY) + 0.5) * dy
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords) # X_grid: (NY, NX)

    save_folder = './cache/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 1. Observation system
    plt.figure(figsize=(8, 6))
    plt.scatter(X_grid.flatten(), Y_grid.flatten(), c='blue', marker='v', label='MT Stations')
    plt.title('3D MT Observation System (Surface Stations)')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '01_observation_system.png'), dpi=300)
    print(f"-> Saved: {os.path.join(save_folder, '01_observation_system.png')}")
    plt.close()

    # 2. Sounding curves
    center_x_idx = NX // 2
    center_y_idx = NY // 2
    
    f_station = freqs_np
    rxy_station = app_res[:, center_y_idx, center_x_idx, 0]
    ryx_station = app_res[:, center_y_idx, center_x_idx, 1]
    
    phi_xy_station = np.abs(phase[:, center_y_idx, center_x_idx, 0])
    phi_yx_station_raw = phase[:, center_y_idx, center_x_idx, 1]
    
    phi_yx_station = phi_yx_station_raw + 180.0
    phi_yx_station = phi_yx_station % 360
    phi_yx_station = np.where(phi_yx_station > 180, phi_yx_station - 360, phi_yx_station)
    phi_yx_station = np.abs(phi_yx_station)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    ax1.loglog(f_station, rxy_station, 'o-', color='red', label='$\\rho_{xy}$ (TE)')
    ax1.loglog(f_station, ryx_station, 's-', color='blue', label='$\\rho_{yx}$ (TM)')
    ax1.set_ylabel('Apparent Resistivity ($\\Omega \\cdot m$)')
    ax1.set_title(f'MT Sounding Curves at Center Station (X={x_coords[center_x_idx]}m, Y={y_coords[center_y_idx]}m)')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.invert_xaxis()

    ax2.semilogx(f_station, phi_xy_station, 'o-', color='red', label='$\\phi_{xy}$ (TE)')
    ax2.semilogx(f_station, phi_yx_station, 's-', color='blue', label='$\\phi_{yx}$ (TM)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (Degrees)')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '02_sounding_curves.png'), dpi=300)
    print(f"-> Saved: {os.path.join(save_folder, '02_sounding_curves.png')}")
    plt.close()

    # 3. 2D Map Slice
    freq_idx = len(freqs_np) // 2
    freq_to_plot = freqs_np[freq_idx]

    RhoXY_grid = app_res[freq_idx, :, :, 0]
    RhoYX_grid = app_res[freq_idx, :, :, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    im1 = ax1.pcolormesh(X_grid, Y_grid, RhoXY_grid, shading='nearest', cmap='jet')
    fig.colorbar(im1, ax=ax1, label='Apparent Resistivity $\\rho_{xy}$ (TE) ($\\Omega \\cdot m$)')
    ax1.scatter(X_grid.flatten(), Y_grid.flatten(), c='black', marker='.', s=10, alpha=0.5)
    ax1.set_title(f'$\\rho_{{xy}}$ (TE) at {freq_to_plot:.3f} Hz')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.axis('equal')

    im2 = ax2.pcolormesh(X_grid, Y_grid, RhoYX_grid, shading='nearest', cmap='jet')
    fig.colorbar(im2, ax=ax2, label='Apparent Resistivity $\\rho_{yx}$ (TM) ($\\Omega \\cdot m$)')
    ax2.scatter(X_grid.flatten(), Y_grid.flatten(), c='black', marker='.', s=10, alpha=0.5)
    ax2.set_title(f'$\\rho_{{yx}}$ (TM) at {freq_to_plot:.3f} Hz')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    ax2.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '03_apparent_res_map.png'), dpi=300)
    print(f"-> Saved: {os.path.join(save_folder, '03_apparent_res_map.png')}")
    plt.close()

    # 4. Pseudosection
    rho_xy_section = app_res[:, center_y_idx, :, 0]
    rho_yx_section = app_res[:, center_y_idx, :, 1]
    
    phi_xy_section = np.abs(phase[:, center_y_idx, :, 0])
    phi_yx_section_raw = phase[:, center_y_idx, :, 1]
    phi_yx_section = phi_yx_section_raw + 180.0
    phi_yx_section = phi_yx_section % 360
    phi_yx_section = np.where(phi_yx_section > 180, phi_yx_section - 360, phi_yx_section)
    phi_yx_section = np.abs(phi_yx_section)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    X_mesh, F_mesh = np.meshgrid(x_coords, freqs_np)

    im1 = axes[0, 0].pcolormesh(X_mesh, F_mesh, np.log10(rho_xy_section), shading='nearest', cmap='jet')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_title(f'$\\log_{{10}}\\rho_{{xy}}$ (TE) at Y = {center_y_idx * dy + dy/2} m')
    fig.colorbar(im1, ax=axes[0, 0], label='Log10($\\rho$) ($\\Omega \\cdot m$)')

    im2 = axes[0, 1].pcolormesh(X_mesh, F_mesh, np.log10(rho_yx_section), shading='nearest', cmap='jet')
    axes[0, 1].set_title(f'$\\log_{{10}}\\rho_{{yx}}$ (TM) at Y = {center_y_idx * dy + dy/2} m')
    fig.colorbar(im2, ax=axes[0, 1], label='Log10($\\rho$) ($\\Omega \\cdot m$)')

    im3 = axes[1, 0].pcolormesh(X_mesh, F_mesh, phi_xy_section, shading='nearest', cmap='jet')
    axes[1, 0].set_xlabel('X Coordinate (m)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_title(f'Phase $\\phi_{{xy}}$ (TE) at Y = {center_y_idx * dy + dy/2} m')
    fig.colorbar(im3, ax=axes[1, 0], label='Phase (Degrees)')

    im4 = axes[1, 1].pcolormesh(X_mesh, F_mesh, phi_yx_section, shading='nearest', cmap='jet')
    axes[1, 1].set_xlabel('X Coordinate (m)')
    axes[1, 1].set_title(f'Phase $\\phi_{{yx}}$ (TM) at Y = {center_y_idx * dy + dy/2} m')
    fig.colorbar(im4, ax=axes[1, 1], label='Phase (Degrees)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, '04_pseudosection.png'), dpi=300)
    print(f"-> Saved: {os.path.join(save_folder, '04_pseudosection.png')}")
    plt.close()

    # 5. True Resistivity Map
    try:
        model_3d = data.reshape((NZ, NY, NX))  # From previous open()
        true_section = model_3d[:, center_y_idx, :]
        
        plt.figure(figsize=(10, 4))
        z_1d = (np.arange(NZ) + 0.5) * dz
        X_mesh_t, Z_mesh_t = np.meshgrid(x_coords, z_1d)
        
        im3 = plt.pcolormesh(X_mesh_t, Z_mesh_t, np.log10(true_section), shading='nearest', cmap='jet')
        plt.gca().invert_yaxis()
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Depth Z (m)')
        plt.title(f'True Resistivity Model ($\\log_{{10}}\\rho$) at Y = {center_y_idx * dy + dy/2} m')
        plt.colorbar(im3, label='Log10(True Res.) ($\\Omega \\cdot m$)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, '05_true_model_section.png'), dpi=300)
        print(f"-> Saved: {os.path.join(save_folder, '05_true_model_section.png')}")
        plt.close()
    except Exception as e:
        print(f"Error plotting true model: {e}")

    print("All visualizations complete! Check the './cache/' directory for PNGs.")
    # save data for 3D visualization
    np.savetxt(os.path.join(save_folder, 'apparent_res_3d.txt'), app_res.reshape(len(freqs_np), NY, NX, 2).reshape(len(freqs_np), -1))
    np.savetxt(os.path.join(save_folder, 'phase_3d.txt'), phase.reshape(len(freqs_np), NY, NX, 2).reshape(len(freqs_np), -1))
    print(f"-> Saved 3D data for visualization: {os.path.join(save_folder, 'apparent_res_3d.txt')}, {os.path.join(save_folder, 'phase_3d.txt')}")
