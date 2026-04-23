import torch
import numpy as np
from mt_forward import MTForward3D
import time

filepath = '/home/wangyh/Project/MTForward3D/MTModel/em_model_3d_50x30x50.bin'
NX, NY, NZ = 50, 30, 50
dx, dy, dz = 160.0, 160.0, 80.0
with open(filepath, 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
rho_tensor = torch.from_numpy(data.astype(np.float64)).view(NZ, NY, NX).permute(2, 1, 0).contiguous()

# Test 4 frequencies
freqs = [100.0, 10.0, 1.0, 0.1]
operator = MTForward3D(freqs, dx, dy, dz)

print("--- 1 GPU, 1 Thread (Ideal single GPU) ---")
start = time.time()
operator(rho_tensor, num_threads=1, device_ids=[0])
print(f"Time: {time.time() - start:.2f}s\n")

print("--- 1 GPU, 4 Threads (Oversubscribed) ---")
start = time.time()
operator(rho_tensor, num_threads=4, device_ids=[0])
print(f"Time: {time.time() - start:.2f}s\n")

num_devices = torch.cuda.device_count()
if num_devices > 1:
    print(f"--- {num_devices} GPUs, {num_devices} Threads (Ideal Multi-GPU) ---")
    start = time.time()
    operator(rho_tensor, num_threads=num_devices, device_ids=list(range(num_devices)))
    print(f"Time: {time.time() - start:.2f}s\n")
