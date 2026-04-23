import torch
from mt_forward import MTForward3D
import time

if __name__ == "__main__":
    # Small test on uniform box to demonstrate timing and explicit Multi-GPU usage
    NX, NY, NZ = 16, 16, 16
    dx, dy, dz = 100.0, 100.0, 100.0
    
    # 8 frequencies to demonstrate load balancing
    freqs = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    rho = torch.ones((NX, NY, NZ)) * 100.0 
    operator = MTForward3D(freqs, dx, dy, dz)
    
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    
    # Example 1: Use specific explicitly defined GPUs -> Notice device_ids argument
    my_gpus = [1] if torch.cuda.device_count() > 1 else [0]
    print(f"\n--- Test 1: Forcing run on strictly GPU(s) {my_gpus} ---")
    start = time.time()
    app_res, phase = operator(rho, num_threads=4, device_ids=my_gpus)
    print(f"Time taken (Restricted to GPU {my_gpus}): {time.time() - start:.2f} s")

    # Example 2: Use All valid GPUs
    all_gpus = list(range(torch.cuda.device_count()))
    print(f"\n--- Test 2: Distributed dynamically across all GPU(s) {all_gpus} ---")
    start = time.time()
    app_res, phase = operator(rho, num_threads=4, device_ids=all_gpus)
    print(f"Time taken (All GPUs): {time.time() - start:.2f} s")
