import torch
from mt_forward import MTForward3D
import time

if __name__ == "__main__":
    # Small identical test on uniform box to demonstrate timing and multi-GPU usage
    NX, NY, NZ = 16, 16, 16
    dx, dy, dz = 100.0, 100.0, 100.0
    
    # 8 frequencies to demonstrate load balancing across GPUs
    freqs = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    rho = torch.ones((NX, NY, NZ)) * 100.0 
    operator = MTForward3D(freqs, dx, dy, dz)
    
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    print("Testing parallel OpenMP + Multi-GPU compute over 8 frequencies:")
    start = time.time()
    
    # Use 4 OMP threads to overlap freq solves out to your available GPUs automatically.
    app_res, phase = operator(rho, num_threads=4)
    end = time.time()

    print(f"Time taken to solve 8 frequencies: {end - start:.2f} s")
