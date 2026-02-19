import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import scipy.io as sio
from glob import glob
import time
import gc  # 垃圾回收

# simpeg 核心库 (新版API)
import discretize
from simpeg import maps
import simpeg.electromagnetics.natural_source as nsem
from simpeg.electromagnetics.natural_source import receivers, sources

def create_mesh_3d(core_shape, core_cell_size, padding_distance, air_layer_height):
    """
    创建 3D TensorMesh，包含核心区、扩边区和空气层
    """
    ncx, ncy, ncz = core_shape
    dx, dy, dz = core_cell_size
    
    # 1. 核心网格 (Core Mesh)
    hx = [(dx, ncx)]
    hy = [(dy, ncy)]
    hz = [(dz, ncz)]
    
    # 2. 扩边 (Padding) - 向外指数延展以满足边界条件
    # pad_rate: 扩边系数, npad: 扩边层数
    npad_x, npad_y, npad_z = 5, 5, 5
    pad_rate = 1.3
    
    # 直接使用 unpack_widths 构建带扩边的网格
    hx = discretize.utils.unpack_widths([(dx, npad_x, -pad_rate), (dx, ncx), (dx, npad_x, pad_rate)])
    hy = discretize.utils.unpack_widths([(dy, npad_y, -pad_rate), (dy, ncy), (dy, npad_y, pad_rate)])
    
    # 垂直方向：地下向下扩边，地上添加空气层
    # 这里的 mesh z 轴通常定义为向上为正，地表为 0
    # 我们需要构建一个从深部(-z)到空气层(+z)的向量
    
    # 地下扩边
    hz_sub = discretize.utils.unpack_widths([(dz, npad_z, -pad_rate), (dz, ncz)])
    # 空气层 (通常需要很厚，至少几个趋肤深度)
    hz_air = discretize.utils.unpack_widths([(dz, npad_z, pad_rate)])
    
    hz_total = np.r_[hz_sub, hz_air]
    
    # 创建网格
    mesh = discretize.TensorMesh([hx, hy, hz_total], x0='CCC') # x0='CCC' 表示中心对齐
    
    # 调整 z 轴零点到地表 (核心区域顶部)
    # 计算核心区域顶部的索引
    # 简单做法：将 mesh 的 z 原点移动，使得空气层底界面位于 z=0
    # mesh.origin 默认为 'CCC' 会把整体中心放在 0，这里需要手动调整 z
    z_origin = -np.sum(hz_sub) 
    x_origin = -np.sum(hx) / 2
    y_origin = -np.sum(hy) / 2
    
    mesh.origin = np.r_[x_origin, y_origin, z_origin]
    
    return mesh

def run_simpeg_mt():
    # ---------------- 1. 参数设置 ----------------
    # 核心区域
    NX, NY, NZ_sub = 16, 16, 15 
    # 增大网格间距以覆盖更大深度范围
    DX, DY, DZ = 500.0, 500.0, 300.0  # 单位：米
    
    # 频率 (Hz) - 使用更低的频率以探测更深的地层
    # 趋肤深度 δ ≈ 503√(ρ/f) 米
    # 例如：0.01 Hz, 100 Ω·m → δ ≈ 50 km
    #       0.1 Hz, 100 Ω·m → δ ≈ 16 km
    #       1 Hz, 100 Ω·m → δ ≈ 5 km
    freqs = np.logspace(-2, 1, 10)  # 0.01Hz, 0.1Hz, 1Hz, 10Hz
    
    input_dir = 'em_models'
    output_dir = 'em_results_simpeg'
    os.makedirs(output_dir, exist_ok=True)
    
    npz_files = sorted(glob(os.path.join(input_dir, '*.npz')))
    
    if not npz_files:
        print("未找到模型文件，请检查 em_models 文件夹")
        return

    # ---------------- 2. 构建模拟网格 (只需一次) ----------------
    print("正在构建 3D 网格...")
    mesh = create_mesh_3d((NX, NY, NZ_sub), (DX, DY, DZ), 5000, 5000)
    print(f"网格总单元数: {mesh.nC} (nx={mesh.shape_cells[0]}, ny={mesh.shape_cells[1]}, nz={mesh.shape_cells[2]})")
    
    # 识别空气层和地下层
    # mesh.gridCC[:, 2] 是所有单元中心的 z 坐标
    # z > 0 为空气 (假设地表在 z=0)
    air_indices = mesh.gridCC[:, 2] > 0
    subsurface_indices = ~air_indices
    
    # ---------------- 3. 定义接收机 (Survey) ----------------
    # 在地表定义接收机阵列
    # 我们只在核心区域的中心放接收机
    rx_x = np.linspace(-NX*DX/2 + DX/2, NX*DX/2 - DX/2, NX)
    rx_y = np.linspace(-NY*DY/2 + DY/2, NY*DY/2 - DY/2, NY)
    RX_X, RX_Y = np.meshgrid(rx_x, rx_y, indexing='ij')  # 使用'ij'索引确保顺序正确
    rx_locs = np.c_[RX_X.flatten(), RX_Y.flatten(), np.zeros(NX*NY)] # z=0 at surface
    
    # 定义要测量的分量: Zxy (实/虚), Zyx (实/虚)
    # simpeg 新版 API 使用 Impedance 替代 PointNaturalSource
    rx_list = [
        receivers.Impedance(rx_locs, orientation="xy", component="real"),
        receivers.Impedance(rx_locs, orientation="xy", component="imag"),
        receivers.Impedance(rx_locs, orientation="yx", component="real"),
        receivers.Impedance(rx_locs, orientation="yx", component="imag"),
    ]
    
    # 定义源列表 (对每个频率定义两个极化方向的平面波)
    # 3D MT 需要两个极化方向: x极化和y极化
    src_list = []
    for freq in freqs:
        # 必须指定极化方向：'X' 或 'Y'
        src_list.append(sources.PlanewaveXYPrimary(rx_list, freq))
        
    survey = nsem.Survey(src_list)

    # ---------------- 4. 只处理第一个模型 ----------------
    # 只取第一个模型进行测试
    npz_files = npz_files[:1]
    
    for file_path in npz_files:
        filename = os.path.basename(file_path)
        output_name = filename.replace('.npz', '_simpeg.mat')
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\n正在处理: {filename}")
        
        # --- A. 读取并映射模型 ---
        try:
            data = np.load(file_path)
            # 读取电阻率数据 - 使用float32节省内存
            if 'resistivity' in data:
                model_raw = np.array(data['resistivity'], dtype=np.float32)  # 使用float32减少内存
            else:
                print(f"  未找到 resistivity 数据，跳过")
                continue
            
            print(f"  原始模型形状: {model_raw.shape}, 类型: {model_raw.dtype}")
            
            # 下采样到网格尺寸 (NX, NY, NZ_sub)
            # 使用 scipy.ndimage.zoom 进行重采样
            from scipy.ndimage import zoom
            
            # 计算缩放因子
            target_shape = (NX, NY, NZ_sub)
            
            # 如果原始模型是 (z, y, x) 格式，先转置
            if model_raw.shape[0] < model_raw.shape[1]:
                # 假设是 (z, y, x) 格式，转为 (x, y, z)
                model_raw = model_raw.transpose(2, 1, 0)
            
            zoom_factors = (
                target_shape[0] / model_raw.shape[0],
                target_shape[1] / model_raw.shape[1],
                target_shape[2] / model_raw.shape[2]
            )
            
            model_core = zoom(model_raw, zoom_factors, order=1)  # 双线性插值
            print(f"  下采样后形状: {model_core.shape}")
            
            # 释放原始模型内存
            del model_raw, data
            gc.collect()
            
            # 确保电阻率为正值并转换为float32
            model_core = model_core.astype(np.float32)
            model_core[model_core < 0.1] = 0.1
                
        except Exception as e:
            print(f"读取模型失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # --- B. 构建全网格电导率模型 (Sigma) ---
        # 1. 初始化全场模型为背景值 (例如 100 ohm-m) - 使用float32
        sigma_model = np.ones(mesh.nC, dtype=np.float32) * (1.0 / 100.0)
        
        # 2. 填充空气层 (1e-8 S/m)
        sigma_model[air_indices] = 1e-8
        
        # 3. 填充核心区域地下模型
        # 我们需要找到核心区域在全网格中的索引位置
        # 这是一个稍微复杂的一步，为了简化，我们假设核心模型填在网格几何中心下部
        # 更好的方法是使用 InjectActiveCells，但这里我们手动映射以确保准确
        
        # 找到核心区域的 x, y, z 范围索引
        # 注意：这里简化处理，假设 mesh 构建时核心就在中间
        # SimPEG 的 TensorMesh 内部存储顺序是 x 变化最快，然后 y，然后 z
        
        # 为了不出错，最稳健的方法是使用 utils.model_builder
        # 但既然我们有 core model，我们创建一个 Active Mapping
        
        # 这里使用一个简单策略：
        # 我们知道核心网格是规则的。我们直接把 core model 嵌入到 sigma_model 中
        # 需要计算核心部分在全网格中的切片索引
        
        # 实际上，SimPEG 的模型是一个长向量。
        # 最简单的方法：遍历所有单元中心，如果在核心范围内，就赋值
        # (这种方法在 Python 中慢，但逻辑最清晰)
        
        core_x_lim = [-NX*DX/2, NX*DX/2]
        core_y_lim = [-NY*DY/2, NY*DY/2]
        core_z_lim = [-NZ_sub*DZ, 0] # 假设地表为0，地下延伸到负深
        
        cc = mesh.gridCC # 所有单元中心坐标 (nC, 3)
        
        in_core = (
            (cc[:, 0] >= core_x_lim[0]) & (cc[:, 0] < core_x_lim[1]) &
            (cc[:, 1] >= core_y_lim[0]) & (cc[:, 1] < core_y_lim[1]) &
            (cc[:, 2] >= core_z_lim[0]) & (cc[:, 2] < 0)
        )
        
        # 赋值核心电导率 - 使用插值确保正确映射
        from scipy.interpolate import RegularGridInterpolator
        
        # 创建核心模型的坐标网格
        core_x = np.linspace(core_x_lim[0] + DX/2, core_x_lim[1] - DX/2, NX)
        core_y = np.linspace(core_y_lim[0] + DY/2, core_y_lim[1] - DY/2, NY)
        core_z = np.linspace(core_z_lim[0] + DZ/2, -DZ/2, NZ_sub)  # z从深到浅
        
        # 创建插值器 (model_core 形状应为 (NX, NY, NZ_sub))
        interp_func = RegularGridInterpolator(
            (core_x, core_y, core_z), 
            model_core, 
            method='linear',
            bounds_error=False,
            fill_value=100.0  # 边界外用背景电阻率
        )
        
        # 对核心区域内的点进行插值
        core_points = cc[in_core]
        core_rho = interp_func(core_points)
        sigma_model[in_core] = 1.0 / core_rho
        
        # --- C. 设置模拟 (Simulation) ---
        # 使用 Primary-Secondary 方法 (NSEM 标准)
        # 需要一个背景模型 (1D background)，通常设为均匀半空间或层状
        sigma_background = np.ones(mesh.nC, dtype=np.float32) * (1.0 / 100.0)
        sigma_background[air_indices] = 1e-8
        
        # 定义映射
        model_map = maps.IdentityMap(nP=mesh.nC)
        
        # 使用 CPU 求解器（不需要 pydiso）
        from simpeg import SolverLU
        solver_to_use = SolverLU
        print("  使用 SolverLU 求解器")
        
        sim = nsem.simulation.Simulation3DPrimarySecondary(
            mesh,
            survey=survey,
            sigmaMap=model_map,
            sigmaPrimary=sigma_background,
            solver=solver_to_use,  # 使用迭代求解器（不需要pydiso）
        )
        
        print("  开始正演计算 (这可能需要几分钟)...")
        t0 = time.time()
        
        # 运行正演
        try:
            dpred = sim.dpred(sigma_model)
            print(f"  计算完成，耗时: {time.time()-t0:.2f} 秒")
        except Exception as e:
            print(f"  SimPEG 计算出错: {e}")
            import traceback
            traceback.print_exc()
            # 清理内存后继续
            del sim, sigma_model, sigma_background
            gc.collect()
            continue
            
        # --- D. 后处理: 阻抗 -> 视电阻率/相位 ---
        # dpred 的结果是一个长向量，顺序对应 rx_list 中的定义
        # 顺序: [Freq1_RxAll_Zxy_Real, Freq1_RxAll_Zxy_Imag, ..., FreqN_...]
        
        n_rx = len(rx_locs)
        n_freq = len(freqs)
        
        # 重塑数据: (n_freq, n_component, n_rx)
        # component order in list: Zxy_r, Zxy_i, Zyx_r, Zyx_i
        data_reshaped = dpred.reshape((n_freq, 4, n_rx))
        
        Zxy = data_reshaped[:, 0, :] + 1j * data_reshaped[:, 1, :]
        Zyx = data_reshaped[:, 2, :] + 1j * data_reshaped[:, 3, :]
        
        # 转置为 (n_rx, n_freq) 以匹配之前的格式
        Zxy = Zxy.T
        Zyx = Zyx.T
        
        mu = 4e-7 * np.pi
        
        # 计算视电阻率 (Rho_a) 和 相位 (Phase)
        # Rho_xy = |Zxy|^2 / (omega * mu)
        # Phase_xy = atan2(imag, real)
        
        rho_xy = np.zeros((n_rx, n_freq))
        phs_xy = np.zeros((n_rx, n_freq))
        rho_yx = np.zeros((n_rx, n_freq))
        phs_yx = np.zeros((n_rx, n_freq))
        
        for i_f, f in enumerate(freqs):
            omega = 2 * np.pi * f
            
            # XY Mode
            z_mag_sq = np.abs(Zxy[:, i_f])**2
            rho_xy[:, i_f] = z_mag_sq / (omega * mu)
            phs_xy[:, i_f] = np.degrees(np.arctan2(Zxy[:, i_f].imag, Zxy[:, i_f].real))
            
            # YX Mode
            z_mag_sq = np.abs(Zyx[:, i_f])**2
            rho_yx[:, i_f] = z_mag_sq / (omega * mu)
            phs_yx[:, i_f] = np.degrees(np.arctan2(Zyx[:, i_f].imag, Zyx[:, i_f].real))
            
        # 整理为 3D 数组以便保存 (NX, NY, n_freq)
        # 注意：由于使用了 indexing='ij'，现在顺序是正确的
        rho_xy_3d = rho_xy.reshape(NX, NY, n_freq)
        phs_xy_3d = phs_xy.reshape(NX, NY, n_freq)
        rho_yx_3d = rho_yx.reshape(NX, NY, n_freq)
        phs_yx_3d = phs_yx.reshape(NX, NY, n_freq)
        
        # 保存
        sio.savemat(output_path, {
            'rho_xy': rho_xy_3d,
            'phs_xy': phs_xy_3d,
            'rho_yx': rho_yx_3d,
            'phs_yx': phs_yx_3d,
            'freqs': freqs,
            'model_core': model_core # 保存原始模型以备查
        })
        print(f"  结果已保存: {output_name}")
        
        # 清理内存
        del sim, sigma_model, sigma_background, dpred, Zxy, Zyx
        del rho_xy, rho_yx, phs_xy, phs_yx
        del rho_xy_3d, rho_yx_3d, phs_xy_3d, phs_yx_3d, model_core
        gc.collect()
        print(f"  内存已清理")

if __name__ == "__main__":
    run_simpeg_mt()