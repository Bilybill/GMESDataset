import torch
import numpy as np
import time
from mt_forward import MTForward3D
filepath = '/home/wangyh/Project/MTForward3D/MTModel/em_model_3d_50x30x50.bin'
NX, NY, NZ = 50, 30, 50
dx, dy, dz = 160.0, 160.0, 80.0
operator = MTForward3D([1.0], dx, dy, dz)
data = np.fromfile(filepath, dtype=np.float32)
rho_tensor = torch.from_numpy(data.astype(np.float64)).view(NZ, NY, NX).permute(2, 1, 0).contiguous()
start = time.time()
operator(rho_tensor)
print(f"Time: {time.time()-start:.2f} s")
