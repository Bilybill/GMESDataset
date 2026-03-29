# 正演结果输出说明

`run_multiphysics_forward.py` 运行完成后，所有的物理场以及中间生成的模型数据均会保存在 `--save_dir` 目录下（默认 `DATAFOLDER/Cache/ForwardOutput`）。

## 生成的文件：
1. **地质属性模型 (*.npz)**:
   - `forward_models.npz`: 含有键 `vp`, `rho`, `res`, `chi` (三维数组) 和标量 `dx`, `dy`, `dz`。其中 `rho` 统一使用 `kg/m^3`。用于后期的 3D 绘图 (如用 cigvis 重新制图)。

2. **重力和磁力 (*.npy)**:
   - `forward_gravity.npy`: 二维数组 (nx, ny)，单位 mGal。
   - `forward_magnetic.npy`: 二维数组 (nx, ny)，单位 nT。

3. **电磁 (MT)**:
   - `forward_mt_app_res.npy`: 视电阻率张量，如 4 维 (N_freq, nx, ny, 2 [ex/ey])。
   - `forward_mt_phase.npy`: 相位张量。

4. **地震 (*.npy)**:
   - `forward_seismic.npy`: 2D slice 共炮点记录记录，形如 (N_shot, N_receiver, N_time)。

## 怎么加载作图？
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取重力
grav = np.load("DATAFOLDER/Cache/ForwardOutput/forward_gravity.npy")
plt.imshow(grav.T, origin='lower', cmap='jet')
plt.show()

# 读取模型
models = np.load("DATAFOLDER/Cache/ForwardOutput/forward_models.npz")
vp = models['vp']
#...
```
