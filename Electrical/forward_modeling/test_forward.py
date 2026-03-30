import torch
import numpy as np
from mt_forward import MTForward3D
import time
import os

if __name__ == "__main__":
    filepath = '/home/wangyh/Project/GMESUni/MTForward3D/MTModel/em_model_3d_50x30x50.bin'
    
    NX, NY, NZ = 50, 30, 50
    dx, dy, dz = 160.0, 160.0, 80.0

    print(f"Loading binary model from {filepath}")
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    if len(data) != NX * NY * NZ:
        raise ValueError(f"Size mismatch: expected {NX*NY*NZ}, got {len(data)}")

    # We map the flat array cleanly into PyTorch, ensuring memory mapping conforms
    # to what our updated C++ wrapper expects.
    rho_tensor = torch.from_numpy(data.astype(np.float64)).view(NZ, NY, NX).permute(2, 1, 0).contiguous()

    print("Initializing PyTorch Operator...")
    operator = MTForward3D(None, dx, dy, dz)
    
    print("Running forward modeling with auto-selected frequencies...")
    start = time.time()
    # It passes the large model straight to device via the wrapper
    app_res, phase = operator(rho_tensor)
    end = time.time()
    freqs = list(operator.last_freqs)

    print(f"\n✅ Forward modeling complete! Total Operator Time: {end - start:.2f} s")
    print(f"Auto-selected Frequencies ({len(freqs)}): {freqs}")
    print(f"Apparent Resistivity Tensor Shape: {tuple(app_res.shape)}   (n_freqs, NX, NY, 2)")
    print(f"Phase Tensor Shape: {tuple(phase.shape)}")
    
    # Quick sanity check printout
    print(f"Example Target - Highest Freq ({freqs[0]} Hz), Center (X={NX//2}, Y={NY//2}):")
    print(f"  Apparent Resistivity (Zxy, Zyx):", app_res[0, NX//2, NY//2].numpy())
    print(f"  Phase (Zxy, Zyx):", phase[0, NX//2, NY//2].numpy())
