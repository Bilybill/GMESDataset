import torch
import numpy as np
from mt_forward import MTForward3D
import time

if __name__ == "__main__":
    filepath = '/home/wangyh/Project/MTForward3D/MTModel/em_model_3d_50x30x50.bin'
    
    NX, NY, NZ = 50, 30, 50
    dx, dy, dz = 160.0, 160.0, 80.0
    f_min, f_max = 0.01, 100.0

    print(f"Loading binary model from {filepath}")
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

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
    
    num_devices = torch.cuda.device_count()
    print(f"System has {num_devices} CUDA devices available.\n")

    # ==========================================
    # 1. Benchmark Single GPU (Force GPU 0)
    # ==========================================
    print("====================================")
    print("🚀 Test 1: Single GPU Execution")
    print("====================================")
    start_single = time.time()
    # using 4 threads on a single GPU to keep its queue saturated
    app_res_single, phase_single = operator(rho_tensor, num_threads=1, device_ids=[0])
    end_single = time.time()
    time_single = end_single - start_single
    print(f"✅ Single GPU Time: {time_single:.2f} s")

    # ==========================================
    # 2. Benchmark Multi-GPU (All Available)
    # ==========================================
    print("\n====================================")
    print(f"🚀 Test 2: Multi-GPU Execution ({num_devices} GPUs)")
    print("====================================")
    start_multi = time.time()
    # Distribute across all GPUs using exactly 4 threads per GPU ideally, but we'll spawn 8 threads total
    all_gpus = list(range(num_devices))
    app_res_multi, phase_multi = operator(rho_tensor, num_threads=len(all_gpus), device_ids=all_gpus)
    end_multi = time.time()
    time_multi = end_multi - start_multi
    print(f"✅ Multi-GPU Time ({num_devices} GPUs): {time_multi:.2f} s")
    
    print("\n====================================")
    print("📊 BENCHMARK RESULTS")
    print("====================================")
    print(f"Number of frequencies: {len(freqs)}")
    print(f"Grid size: {NX}x{NY}x{NZ}")
    print(f"Single GPU Time  : {time_single:.2f} s")
    print(f"Multi GPU Time   : {time_multi:.2f} s")
    if time_multi > 0:
        speedup = time_single / time_multi
        print(f"Speedup Factor   : {speedup:.2f}x")
    print("====================================")
