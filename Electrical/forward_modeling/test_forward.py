import torch
import numpy as np
from mt_forward import MTForward3D
import time
import os

if __name__ == "__main__":
    filepath = '/home/wangyh/Project/MTForward3D/MTModel/em_model_3d_50x30x50.bin'
    
    # Parameters mirroring test_forward.sh
    # ./emforward3d <file> NX NY NZ dx dy dz f_min f_max
    NX, NY, NZ = 50, 30, 50
    dx, dy, dz = 160.0, 160.0, 80.0
    f_min, f_max = 0.01, 100.0

    print(f"Loading binary model from {filepath}")
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    if len(data) != NX * NY * NZ:
        raise ValueError(f"Size mismatch: expected {NX*NY*NZ}, got {len(data)}")

    # We map the flat array cleanly into PyTorch, ensuring memory mapping conforms
    # to what our updated C++ wrapper expects.
    rho_tensor = torch.from_numpy(data.astype(np.float64)).view(NZ, NY, NX).permute(2, 1, 0).contiguous()

    # Calculate frequencies as emforward3d.cpp does
    phoenix_coeffs = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]
    max_power = int(np.ceil(np.log10(f_max)))
    min_power = int(np.floor(np.log10(f_min)))
    
    freqs = []
    for p in range(max_power, min_power - 1, -1):
        power_of_10 = 10.0 ** p
        for coeff in phoenix_coeffs:
            current_f = float(coeff * power_of_10)
            # slight epsilon check for floating point limits
            if current_f <= f_max * 1.0001 and current_f >= f_min * 0.9999:
                freqs.append(current_f)
    if not freqs:
        freqs = [f_max, f_min]
        
    print(f"Calculated Frequencies ({len(freqs)}): {freqs}")

    print("Initializing PyTorch Operator...")
    operator = MTForward3D(freqs, dx, dy, dz)
    
    print("Running forward modeling (This will compute BiCGStab for multi-frequencies)...")
    start = time.time()
    # It passes the large model straight to device via the wrapper
    app_res, phase = operator(rho_tensor)
    end = time.time()

    print(f"\n✅ Forward modeling complete! Total Operator Time: {end - start:.2f} s")
    print(f"Apparent Resistivity Tensor Shape: {tuple(app_res.shape)}   (n_freqs, NY, NX, 2)")
    print(f"Phase Tensor Shape: {tuple(phase.shape)}")
    
    # Quick sanity check printout
    print(f"Example Target - Highest Freq ({freqs[0]} Hz), Center (X={NX//2}, Y={NY//2}):")
    print(f"  Apparent Resistivity (Zxy, Zyx):", app_res[0, NY//2, NX//2].numpy())
    print(f"  Phase (Zxy, Zyx):", phase[0, NY//2, NX//2].numpy())
