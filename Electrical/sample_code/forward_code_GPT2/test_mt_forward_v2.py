import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 确保能找到 forward_code_GPT2/em3d 模块
project_root = '/home/wangyh/Project/GMESDataset/Electrical/forward_code_GPT2'
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入必要模块
from em3d.grid import Grid3D, Receiver
from em3d.forward import simulate_mt_primary_secondary_batch

def plot_3d_model(sigma, grid, receivers=None, title="3D Conductivity Model"):
    """使用 matplotlib 简单切片展示 3D 模型"""
    # 将电导率转为 CPU numpy，取 log10 用于显示
    sig_np = sigma.cpu().numpy()
    nx, ny, nz = sig_np.shape
    
    # 取中心切片
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # XY 切片 (z=cz)
    im0 = axes[0].imshow(np.log10(sig_np[:, :, cz].T), origin='lower', cmap='jet', aspect='auto',
                         extent=[0, nx*grid.dx, 0, ny*grid.dy])
    axes[0].set_title(f'XY Slice (z_idx={cz})')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    
    # XZ 切片 (y=cy)
    im1 = axes[1].imshow(np.log10(sig_np[:, cy, :].T), origin='upper', cmap='jet', aspect='auto',
                         extent=[0, nx*grid.dx, nz*grid.dz, 0])
    axes[1].set_title(f'XZ Slice (y_idx={cy})')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('z (m)')
    
    # YZ 切片 (x=cx)
    im2 = axes[2].imshow(np.log10(sig_np[cx, :, :].T), origin='upper', cmap='jet', aspect='auto',
                         extent=[0, ny*grid.dy, nz*grid.dz, 0])
    axes[2].set_title(f'YZ Slice (x_idx={cx})')
    axes[2].set_xlabel('y (m)')
    axes[2].set_ylabel('z (m)')

    # 可视化接收点 (投影到 XZ 平面)
    if receivers:
        rx_x = [r.x for r in receivers]
        rx_z = [r.z for r in receivers]
        axes[1].scatter(rx_x, rx_z, c='white', marker='v', edgecolors='black', label='Receivers')
        axes[1].legend()

    fig.suptitle(title + " (log10(S/m))")
    plt.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='log10(Conductivity)')
    plt.savefig('model_slices.png', dpi=300)
    print("Model slices saved to model_slices.png")
    plt.show()

def plot_mt_response(receivers, data_list, freqs):
    """绘制视电阻率和相位伪断面图 (Pseudosection)"""
    n_freq = len(freqs)
    n_rx = len(receivers)
    
    # 提取数据用于绘图 (shape: n_freq x n_rx)
    rho_xy = np.zeros((n_freq, n_rx))
    phi_xy = np.zeros((n_freq, n_rx))
    rho_yx = np.zeros((n_freq, n_rx))
    phi_yx = np.zeros((n_freq, n_rx))
    
    rx_locs = np.array([r.x for r in receivers]) # 假设是一维测线沿 x 轴
    
    for i, d in enumerate(data_list):
        rho_xy[i, :] = d['rho_xy'].real.cpu().numpy()
        phi_xy[i, :] = d['phi_xy_deg'].real.cpu().numpy()
        rho_yx[i, :] = d['rho_yx'].real.cpu().numpy()
        phi_yx[i, :] = d['phi_yx_deg'].real.cpu().numpy()

    # 准备绘图网格
    X, Y = np.meshgrid(rx_locs, freqs)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    
    # 辅助绘图函数 - 改用 contourf 以获得更好的填充效果
    def plot_section(ax, data, title, cmap='jet', is_log=True):
        if is_log:
            c = ax.contourf(X, Y, data, 20, cmap=cmap, norm=plt.matplotlib.colors.LogNorm())
        else:
            c = ax.contourf(X, Y, data, 20, cmap=cmap)
            
        ax.set_yscale('log')
        ax.invert_yaxis() # MT 习惯频率从高到低排列（对应深度从浅到深）
        
        # 标记数据点位置
        ax.scatter(X, Y, c='k', s=5, marker='.')
        
        ax.set_title(title)
        fig.colorbar(c, ax=ax)

    # 1. Rho XY
    plot_section(axes[0, 0], rho_xy, 'Apparent Resistivity XY (Ohm-m)', cmap='jet')
    
    # 2. Phase XY
    plot_section(axes[0, 1], phi_xy, 'Phase XY (deg)', cmap='jet', is_log=False)
    
    # 3. Rho YX
    plot_section(axes[1, 0], rho_yx, 'Apparent Resistivity YX (Ohm-m)', cmap='jet')
    
    # 4. Phase YX
    plot_section(axes[1, 1], phi_yx, 'Phase YX (deg)', cmap='jet', is_log=False)

    axes[1, 0].set_xlabel('Receiver X (m)')
    axes[1, 1].set_xlabel('Receiver X (m)')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig('mt_pseudosection.png', dpi=300)
    print("MT Pseudosection saved to mt_pseudosection.png")
    plt.show()

def main():
    # ---------------------------------------------------------
    # 1. 初始化设置
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 定义网格 (和之前类似，但这次我们可以稍微大一点或保持一致)
    nx, ny, nz = 16, 16, 16
    dx, dy, dz = 100.0, 100.0, 50.0 
    
    # 使用复杂双精度
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, 
                  device=device, dtype=torch.complex128)
    
    # ---------------------------------------------------------
    # 2. 构建模型 (含低阻体)
    # ---------------------------------------------------------
    # 背景: 100 ohm-m
    bg_sigma_val = 1.0 / 100.0
    sigma = torch.full(grid.shape_cells, bg_sigma_val, dtype=grid.rdtype, device=grid.device)
    sigma_bg = torch.full_like(sigma, bg_sigma_val)
    
    # 异常体: 10 ohm-m (低阻)
    # 放在中心偏下一点
    c_x, c_y, c_z = nx // 2, ny // 2, nz // 2
    r = 3
    sigma[c_x-r:c_x+r, c_y-r:c_y+r, c_z-r:c_z+r] = 1.0 / 10.0
    
    # 5. 可视化模型
    # 注意: 我们先构造接收点再传入 plot_3d_model 以便一起画出来
    line_y = (ny // 2) * dy + 0.5 * dy
    line_z = 0.5 * dz 
    receivers = []
    # 沿 X 轴布设测线
    for i in range(4, nx - 4):
        x_loc = i * dx + 0.5 * dx
        receivers.append(Receiver(x=x_loc, y=line_y, z=line_z, name=f"Rx_{i}"))
        
    print("Visualizing Model...")
    # 只需要在支持图形界面的环境运行 matplotlib，否则只会保存或不显示
    try:
        plot_3d_model(sigma, grid, receivers)
    except Exception as e:
        print(f"Skipping visualization due to error: {e}")

    # ---------------------------------------------------------
    # 3. 运行正演 (使用第二版 API)
    # ---------------------------------------------------------
    # 增加更多频率以绘制 2D 断面
    freqs = [100.0, 30.0, 10.0, 3.0, 1.0] 
    
    print(f"Starting simulation for frequencies: {freqs} Hz...")
    
    # 第二版的 API 和第一版基本兼容，但内部使用了更强的 Flex 求解器
    # precond='flex' 是第二版默认的，这里显式指定一下
    data_list, info_list = simulate_mt_primary_secondary_batch(
        grid, sigma, sigma_bg, freqs, receivers, 
        tol=1e-6, maxiter=200, restart=20, 
        use_block_secondary=True, # 启用 Block solver
        precond='jacobi'            # 使用 Flexible 混合预条件
    )
    
    # ---------------------------------------------------------
    # 4. 结果展示
    # ---------------------------------------------------------
    print("\nSimulation Complete. Plotting Results...")
    try:
        plot_mt_response(receivers, data_list, freqs)
    except Exception as e:
        print(f"Skipping result plotting due to error: {e}")
        
    # 打印一些文本结果
    f_idx = 0 # 打印 10 Hz 的结果
    print(f"\nSample Data (Freq = {freqs[f_idx]} Hz):")
    d = data_list[f_idx]
    rho_xy = d['rho_xy'].real.cpu().numpy()
    phi_xy = d['phi_xy_deg'].real.cpu().numpy()
    
    print(f"{'Rx_Name':<10} {'Rho_XY':<12} {'Phi_XY':<10}")
    print("-" * 40)
    for i, rx in enumerate(receivers):
        if i % 2 == 0: # 只打印部分以免刷屏
             print(f"{rx.name:<10} {rho_xy[i]:<12.2f} {phi_xy[i]:<10.2f}")

if __name__ == "__main__":
    main()
