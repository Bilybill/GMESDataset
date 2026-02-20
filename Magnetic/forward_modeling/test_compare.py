import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

# Import the GPU function
from mag_forward import forward_mag_tmi

def mat_spatial_prism_magnetic(nx, ny, nz, dx, dy, dz, h, model_slice):
    """
    Python port of `fun_forward_mag.m` to generate ground truth data.
    Uses exact prism integration formula in spatial domain.
    """
    x0 = np.arange(-(nx-1)*dx, (nx-1)*dx + 0.1, dx)
    y0 = np.arange(-(ny-1)*dy, (ny-1)*dy + 0.1, dy)
    z0_vals = np.arange(-(nz*dz+h), -(dz+h) + 0.1, dz)
    
    # Kernel T
    T_exact = np.zeros((2*nx-1, 2*ny-1, nz))
    Ai = [-dx/2, dx/2]
    Bj = [-dy/2, dy/2]
    Ck = [0, dz]
    
    print("Computing Spatial Reference (this might take a few seconds)...")
    for q in range(nz):
        zk_base = z0_vals[q]
        for i_idx, val_i in enumerate(Ai):
            for j_idx, val_j in enumerate(Bj):
                for k_idx, val_k in enumerate(Ck):
                    uijk = (-1)**((i_idx+1) + (j_idx+1) + (k_idx+1))
                    
                    # Compute meshgrid for current shift
                    xi_mesh, yj_mesh = np.meshgrid(x0 + val_i, y0 + val_j, indexing='ij')
                    # Note: indexing='ij' means row=x, col=y. Match output shape (2nx-1, 2ny-1).
                    
                    zk = zk_base + val_k # This is z (negative)
                    rijk = np.sqrt(xi_mesh**2 + yj_mesh**2 + zk**2)
                    
                    # MATLAB: atan(xi*yj/(zk*rijk))
                    # Use np.arctan which matches MATLAB behavior for real numbers
                    with np.errstate(divide="ignore", invalid="ignore"):
                        term = np.arctan(xi_mesh * yj_mesh / (zk * rijk))
                        term = np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    T_exact[:, :, q] += -uijk * term

    D = np.zeros((nx, ny))
    
    # Convolve
    for q in range(nz):
        # MATLAB logic: T layer q corresponds to depth z0[q].
        # z0 array goes from deep to shallow (-1000 to -100).
        # q=0 is deep. q=nz-1 is shallow.
        
        # Model slicing: Standard is k=0 (top) to k=nz-1 (bottom).
        # We need to match the depth.
        # If z0[0] is -1000 (deepest), we should use model slice corresponding to bottom?
        # If model is created with model[... , 2:6] = 1, we assume index 0 is top.
        # So index (nz-1) is bottom.
        
        # We need to map q (deep->shallow) to model (shallow->deep).
        # q=0 (deepest) -> model index nz-1.
        # q=nz-1 (shallowest) -> model index 0.
        
        layer_m = model_slice[:, :, (nz - 1) - q]
        layer_k = T_exact[:, :, q] 
        
        # Valid convolution
        conv_res = convolve2d(layer_k, layer_m, mode='valid')
        D += conv_res
        
    # Apply global scaling from MATLAB "G_T = 10^2 * Ta"
    return D * 100.0

def _normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(arr)) + 1e-12
    return arr / denom


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def compare_results():
    # 1. Setup Model (Same as MATLAB sample)
    nx, ny, nz = 20, 20, 10
    dx, dy, dz = 100., 100., 100.
    
    model = np.zeros((nx, ny, nz))
    # Fill some blocks (near top)
def compare_results():
    # 1. Setup Model (Same as MATLAB sample)
    nx, ny, nz = 20, 20, 10
    dx, dy, dz = 100., 100., 100.
    
    model = np.zeros((nx, ny, nz))
    # Fill some blocks (near top)
    model[8:12, 8:12, 2:6] = 1.0 
    
    # DEBUG: Print non-zero model stats
    print(f"Model non-zero count: {np.count_nonzero(model)}")
    print(f"Model min/max indices: {np.where(model>0)}")

    obs_conf = {"layout": "grid", "n_x": nx, "n_y": ny}

    
    # 2. Run Spatial Reference (MATLAB Port)
    t0 = time.time()
    d_spatial = mat_spatial_prism_magnetic(nx, ny, nz, dx, dy, dz, h=0, model_slice=model)
    print(f"Spatial Calc Time: {time.time()-t0:.2f}s")

    # 3. Run GPU FFT Forward (two modes)
    # MATLAB sample uses M=1 (A/m) and a factor 10^2 = 100, so it is closer to "magnetization" mode.
    # We still report both modes for clarity.
    results = []
    for input_type in ["magnetization", "susceptibility"]:
        t0 = time.time()
        tmi_gpu_t, _ = forward_mag_tmi(
            torch.from_numpy(model).float(), dx, dy, dz,
            heights_m=[0], obs_conf=obs_conf,
            input_type=input_type,
            B0=50000, I_deg=90, A_deg=0,
            output_unit="nt"
        )
        tmi_gpu = tmi_gpu_t[0, 0].cpu().numpy()
        elapsed = time.time() - t0
        results.append((input_type, tmi_gpu, elapsed))

    print("\n--- Comparison Stats ---")
    print(f"Spatial Max (MATLAB port): {d_spatial.max():.4f} nT")
    
    spa_n = _normalize(d_spatial)
    for input_type, tmi_gpu, elapsed in results:
        gpu_n = _normalize(tmi_gpu)
        mae_normal = _mae(gpu_n, spa_n)
        mae_flip_x = _mae(gpu_n[::-1, :], spa_n)
        mae_flip_y = _mae(gpu_n[:, ::-1], spa_n)
        mae_flip_xy = _mae(gpu_n[::-1, ::-1], spa_n)
        ratio = d_spatial.max() / (tmi_gpu.max() + 1e-12)
        
        print(f"\n[Mode: {input_type}] GPU FFT time: {elapsed:.4f}s")
        print(f"GPU FFT Max: {tmi_gpu.max():.4f} nT")
        print(f"Ratio (Spatial/GPU): {ratio:.6f}")
        print(f"MAE Normal: {mae_normal:.4f}")
        print(f"MAE FlipX : {mae_flip_x:.4f}")
        print(f"MAE FlipY : {mae_flip_y:.4f}")
        print(f"MAE FlipXY: {mae_flip_xy:.4f}")

    # Visualize (magnetization mode by default)
    tmi_gpu = results[0][1]
    gpu_n = _normalize(tmi_gpu)
    diff_n = np.abs(gpu_n - spa_n)
    plt.figure(figsize=(15,5))
    plt.subplot(131); im1 = plt.imshow(results[0][1], cmap='jet', origin='lower'); plt.title("GPU FFT (magnetization)"); plt.colorbar(im1)
    plt.subplot(132); im2 = plt.imshow(d_spatial, cmap='jet', origin='lower'); plt.title("MATLAB Port (spatial)"); plt.colorbar(im2)
    plt.subplot(133); im3 = plt.imshow(diff_n, cmap='gray', origin='lower'); plt.title("Diff Norm"); plt.colorbar(im3)
    plt.savefig("comparison_result.png")
    print("Plot saved.")

if __name__ == "__main__":
    compare_results()