import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import cigvis

def load_and_visualize():
    # -----------------------------------------------------------
    # 1. 参数设置
    # -----------------------------------------------------------
    NX, NY, NZ = 50, 30, 50  # 根据 test_model.py 中的定义
    model_file = '/home/wangyh/Project/GMESUni/MTForward3D/MTModel/em_model_3d_50x30x50.bin'
    app_res_file = './cache/apparent_res_3d.txt'
    phase_file = './cache/phase_3d.txt'

    # -----------------------------------------------------------
    # 2. 读取并可视化 3D 电阻率模型 (Real Model)
    # -----------------------------------------------------------
    print(f"正在读取模型文件: {model_file} ...")
    if os.path.exists(model_file):
        # 读取二进制文件 (float32)
        rho = np.fromfile(model_file, dtype=np.float32)
        
        # 检查数据大小是否匹配
        if rho.size != NX * NY * NZ:
            print(f"警告: 模型文件大小 ({rho.size}) 与预期 ({NX}x{NY}x{NZ}) 不符")
            # 尝试根据大小自动推断立方体形状
            n = int(round(rho.size ** (1/3)))
            if n**3 == rho.size:
                rho = rho.reshape(n, n, n)
                print(f"自动重塑为: {n}x{n}x{n}")
            else:
                print("无法重塑数据，跳过 3D 可视化。")
                rho = None
        else:
            # 原始导出的模型维度通常为 (NZ, NY, NX)
            rho = rho.reshape(NZ, NY, NX)
            # cigvis 通常要求数组格式为 (X, Y, Z)，因此这里需要把矩阵转置
            rho = np.transpose(rho, (2, 1, 0))

        if rho is not None:
            print("启动 cigvis 3D 可视化...")
            # 使用 cigvis.create_slices 创建切片节点，更符合 cigvis 标准用法
            # 指定在数据的中心位置创建三个正交切片
            pos_rho = [rho.shape[0]//2, rho.shape[1]//2, rho.shape[2]//2]
            slices = cigvis.create_slices(rho, cmap='jet', pos=pos_rho)
            
            # 添加 colorbar
            # create_colorbar 返回的是单一对象，所以需要用列表包裹后与 slices 列表合并
            cbar = cigvis.create_colorbar(cmap='jet', clim=[rho.min(), rho.max()], label_str='Resistivity (Ohm-m)')
            
            try:
                cigvis.plot3D(slices + [cbar], savename='3D_real_model.png')
            except TypeError:
                 # 如果 cbar 是列表 (虽然可能性小，但以防万一)
                 cigvis.plot3D(slices + cbar, savename='3D_real_model.png')
    else:
        print(f"文件不存在: {model_file}，请先运行 test_model.py 生成模型。")

    # -----------------------------------------------------------
    # 3. 读取并可视化 3D 观测数据体 (Apparent Resistivity & Phase)
    # -----------------------------------------------------------
    if os.path.exists(app_res_file) and os.path.exists(phase_file):
        print("正在读取视电阻率和相位数据...")
        try:
            # 格式: (n_freqs, NY * NX * 2)
            data_res = np.loadtxt(app_res_file)
            data_phs = np.loadtxt(phase_file)

            n_freq = data_res.shape[0]

            print(f"共读取到 {n_freq} 个频点，正在构建伪 3D 数据体 (Freq, Y, X)...")

            # 将 (n_freqs, NY * NX * 2) reshape 为 (n_freqs, NY, NX, 2)
            app_res = data_res.reshape(n_freq, NY, NX, 2)
            phase = data_phs.reshape(n_freq, NY, NX, 2)

            # 分离 TE (xy) 和 TM (yx) 模式
            rho_xy = app_res[..., 0]
            rho_yx = app_res[..., 1]
            phi_xy = phase[..., 0]
            phi_yx = phase[..., 1]

            print("启动观测数据 3D 可视化...")
            print("2x2 图分布: 左上 rho_xy, 右上 rho_yx, 左下 phi_xy, 右下 phi_yx")

            # cigvis 通常接收的格式为 (X, Y, Z)，目前 Z 轴(频率) 在最前面，需要转置 (NX, NY, n_freq)
            rho_xy = np.transpose(rho_xy, (2, 1, 0))
            rho_yx = np.transpose(rho_yx, (2, 1, 0))
            phi_xy = np.transpose(phi_xy, (2, 1, 0))
            phi_yx = np.transpose(phi_yx, (2, 1, 0))

            # 计算指定在数据的中心位置创建三个正交切片的坐标
            pos_vol = [rho_xy.shape[0]//2, rho_xy.shape[1]//2, rho_xy.shape[2]//2]

            # 创建 cigvis 切片节点及 colorbar
            nodes_rxy = cigvis.create_slices(rho_xy, cmap='jet', pos=pos_vol)
            cbar_rxy = cigvis.create_colorbar(cmap='jet', clim=[rho_xy.min(), rho_xy.max()], label_str='App. Res. XY (Ohm-m)')

            nodes_ryx = cigvis.create_slices(rho_yx, cmap='jet', pos=pos_vol)
            cbar_ryx = cigvis.create_colorbar(cmap='jet', clim=[rho_yx.min(), rho_yx.max()], label_str='App. Res. YX (Ohm-m)')

            nodes_pxy = cigvis.create_slices(phi_xy, cmap='jet', pos=pos_vol)
            cbar_pxy = cigvis.create_colorbar(cmap='jet', clim=[phi_xy.min(), phi_xy.max()], label_str='Phase XY (Deg)')

            nodes_pyx = cigvis.create_slices(phi_yx, cmap='jet', pos=pos_vol)
            cbar_pyx = cigvis.create_colorbar(cmap='jet', clim=[phi_yx.min(), phi_yx.max()], label_str='Phase YX (Deg)')

            try:
                vis_nodes = [
                    nodes_rxy + [cbar_rxy], nodes_ryx + [cbar_ryx],
                    nodes_pxy + [cbar_pxy], nodes_pyx + [cbar_pyx]
                ]
                cigvis.plot3D(vis_nodes, grid=(2, 2), share=True, size=(1600, 1200), savename='3D_app_res_phase.png')
            except TypeError:
                # 如果 create_colorbar 返回的是列表
                vis_nodes = [
                    nodes_rxy + cbar_rxy, nodes_ryx + cbar_ryx,
                    nodes_pxy + cbar_pxy, nodes_pyx + cbar_pyx
                ]
                cigvis.plot3D(vis_nodes, grid=(2, 2), share=True, size=(1600, 1200), savename='3D_app_res_phase.png')

        except Exception as e:
            print(f"读取或绘制 3D 观测数据时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("未找到计算结果文件 (.txt)，仅展示模型。请确保已运行 emforward3d 程序。")

if __name__ == '__main__':
    load_and_visualize()