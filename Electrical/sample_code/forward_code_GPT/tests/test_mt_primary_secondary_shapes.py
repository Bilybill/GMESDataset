import sys
import os
import torch
import math

# 假设你的项目路径在 '/home/wangyh/Project/GMESDataset/Electrical/forward_code_GPT'
# 根据实际情况修改下面这行，或者确保你在项目根目录下运行脚本
project_root = '/home/wangyh/Project/GMESDataset/Electrical/forward_code_GPT'
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入必要的模块
from em3d.grid import Grid3D, Receiver
from em3d.forward import simulate_mt_primary_secondary_batch

def main():
    # ---------------------------------------------------------
    # 1. 检查设备 (支持 GPU 加速)
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. 定义 3D 网格
    # ---------------------------------------------------------
    # 这是一个简单的示例网格，实际使用请根据模型尺度调整
    nx, ny, nz = 32, 32, 32
    dx, dy, dz = 100.0, 100.0, 50.0 # 单元尺寸 (米)
    
    # 构造 Grid3D 对象
    # dtype=torch.complex128 (双精度复数) 推荐用于高精度电磁计算
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, 
                  device=device, dtype=torch.complex128)

    print(f"Grid size: {nx}x{ny}x{nz}, Total cells: {nx*ny*nz}")

    # ---------------------------------------------------------
    # 3. 构建电阻率/电导率模型
    # ---------------------------------------------------------
    # 初始化均匀半空间背景 (例如 100 ohm-m -> 0.01 S/m)
    bg_resistivity = 100.0
    bg_sigma_val = 1.0 / bg_resistivity
    
    # sigma 是 3D 张量，形状为 (nx, ny, nz)
    sigma = torch.full(grid.shape_cells, bg_sigma_val, 
                       dtype=grid.rdtype, device=grid.device)
    
    # 设置背景 sigma_bg (用于主场计算)
    sigma_bg = torch.full_like(sigma, bg_sigma_val)

    # --- 添加一个低阻异常体 (例如 10 ohm-m) ---
    anomaly_resistivity = 10.0
    # 在网格中心区域设置异常
    c_x, c_y, c_z = nx // 2, ny // 2, nz // 2
    r = 4 # 半径（网格数）
    sigma[c_x-r:c_x+r, c_y-r:c_y+r, c_z-r:c_z+r] = 1.0 / anomaly_resistivity
    
    # 如果有空气层 (虽然 MT 一般假设都在地下，但如果有地形可以在此设置空气电导率为 1e-8)
    # sigma[:, :, 0:1] = 1e-8 
    # sigma_bg[:, :, 0:1] = 1e-8

    # ---------------------------------------------------------
    # 4. 设置接收点 (Receivers)
    # ---------------------------------------------------------
    # 在地表 (z=0) 附近布设一条测线
    receivers = []
    # 假设测线沿 X 轴，Y 位于中心，Z=0.5 (第一个网格中心)
    line_y = (ny // 2) * dy + 0.5 * dy
    line_z = 0.5 * dz # 位于地表下第一个网格中心
    
    for i in range(4, nx - 4, 2):
        x_loc = i * dx + 0.5 * dx
        # name 主要是为了标记，内部计算用坐标插值
        receivers.append(Receiver(x=x_loc, y=line_y, z=line_z, name=f"Rx_{i}"))

    print(f"Number of receivers: {len(receivers)}")

    # ---------------------------------------------------------
    # 5. 设置频率并运行正演
    # ---------------------------------------------------------
    freqs = [10.0, 1.0, 0.1] # Hz
    
    print("Starting simulation...")
    # simulate_mt_primary_secondary_batch 自动处理两个极化模式并计算视电阻率
    # tol: 求解器收敛容差
    # maxiter: 最大迭代步数
    data_list, info_list = simulate_mt_primary_secondary_batch(
        grid, sigma, sigma_bg, freqs, receivers, 
        tol=1e-6, maxiter=200, restart=20
    )

    # ---------------------------------------------------------
    # 6. 输出结果
    # ---------------------------------------------------------
    # data_list 是一个列表，对应每个频率
    # 每个元素是一个字典，包含各个接收点的 'rho_xy', 'phi_xy', 'rho_yx', etc.
    
    for i, f in enumerate(freqs):
        print(f"\nResults for Frequency: {f} Hz")
        d = data_list[i]
        
        # d['rho_xy'] 是一个 Tensor，长度等于接收点数量
        # 我们可以将其转为 numpy 数组查看
        rho_xy = d['rho_xy'].real.cpu().numpy() # 视电阻率是实数
        phi_xy = d['phase_xy_deg'].real.cpu().numpy()
        rho_yx = d['rho_yx'].real.cpu().numpy()
        phi_yx = d['phase_yx_deg'].real.cpu().numpy()
        
        print(f"{'Rx_ID':<10} {'Rho_XY':<12} {'Phi_XY':<10} {'Rho_YX':<12} {'Phi_YX':<10}")
        print("-" * 60)
        for j, rx in enumerate(receivers):
            print(f"{rx.name:<10} {rho_xy[j]:<12.2f} {phi_xy[j]:<10.2f} {rho_yx[j]:<12.2f} {phi_yx[j]:<10.2f}")

if __name__ == "__main__":
    main()
