import torch
from mt_forward import MTForward3D
import time

if __name__ == "__main__":
    # Small test on a uniform half-space
    NX, NY, NZ = 16, 16, 16
    dx, dy, dz = 100.0, 100.0, 100.0
    freqs = [8.0, 1.0]

    print("Generating tensor...")
    # Uniform 100 Ohm-m model
    rho = torch.ones((NX, NY, NZ)) * 100.0 

    print("Initializing operator...")
    operator = MTForward3D(freqs, dx, dy, dz)
    
    print("Running forward modeling (this may take a few seconds as it calls BiCGStab...)")
    start = time.time()
    app_res, phase = operator(rho)
    end = time.time()

    print(f"Time taken: {end - start:.2f} s")
    print("app_res shape:", app_res.shape)
    print("phase shape:", phase.shape)
    
    # Just print the middle cell result for highest frequency
    mid_x, mid_y = NX // 2, NY // 2
    print(f"Center cell app_res (xy, yx) at {freqs[0]} Hz: {app_res[0, mid_y, mid_x]}")
    print(f"Center cell phase (xy, yx) at {freqs[0]} Hz: {phase[0, mid_y, mid_x]}")
