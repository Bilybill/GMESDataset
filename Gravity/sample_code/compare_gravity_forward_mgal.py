#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare gravity forward outputs in a unified PHYSICAL unit system:
- density unit: kg/m^3
- output unit: mGal (1 mGal = 1e-5 m/s^2)

We compare:
  (1) GraForward (from gra_forward_ori.py): y_ori (dimensionless-ish, missing G and with param = dx*dy*dz/1000)
      -> converted to mGal by multiplying a constant factor:
         y_ori_mgal = y_ori * (G * 1000 / 1e-5) = y_ori * (G * 1e8)

  (2) FFT-based forward that computes physical gz directly:
         gz_si  = G * sum rho * dV * z / r^3
         gz_mgal = gz_si / 1e-5

Important:
- This script is for unit-consistent comparison. The SHAPE/pattern should match if geometry aligns.
- For large sizes, GraForward.set_params() is extremely slow and memory-heavy. Use small in_size unless you have a saved .pth weight.
"""

import os
import argparse
import time
import numpy as np
import torch
import cigvis

G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
SI_TO_MGAL = 1.0 / 1e-5     # (m/s^2) -> mGal

def parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]

def parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip() != ""]

def load_density(path: str):
    """
    Load density model from .npy or .npz. Returns numpy array.
    Supported shapes:
      - (D, H, W)  [preferred]
      - (H, W, D)  [will be transposed to (D, H, W)]
    """
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        for k in ["density", "rho", "model", "data"]:
            if k in z:
                arr = z[k]
                break
        else:
            arr = z[list(z.keys())[0]]
    else:
        raise ValueError(f"Unsupported density file type: {path}")

    if arr.ndim != 3:
        raise ValueError(f"Density must be 3D. Got shape={arr.shape}")
    return arr

def ensure_dhw(arr: np.ndarray, D: int, H: int, W: int):
    """Ensure density array is (D,H,W)."""
    if arr.shape == (D, H, W):
        return arr
    if arr.shape == (H, W, D):
        print("Transposing density from (H,W,D) to (D,H,W)")
        return np.transpose(arr, (2, 0, 1))
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    for p in perms:
        if tuple(arr.shape[i] for i in p) == (D, H, W):
            return np.transpose(arr, p)
    raise ValueError(f"Cannot transpose density to (D,H,W). density.shape={arr.shape}, target={(D,H,W)}")

@torch.no_grad()
def fft_forward_physical_mgal_like_graforward_geometry(
    rho_dhw: torch.Tensor,
    heights: list[float],
    sample: tuple[float,float,float],
    pad_factor: int = 1,
    dtype: torch.dtype = torch.float64,
):
    """
    Physical forward in mGal using the SAME x/y half-cell offset convention as GraForward.set_params().

    Geometry convention matched to GraForward:
      x = (p + 0.5) * dx, where p = m - i in [-(H-1), ..., +(H-1)]
      y = (q + 0.5) * dy, where q = n - j in [-(W-1), ..., +(W-1)]
      z0 = dz*k + height + dz/2, k in [0..D-1]

    Physics:
      gz_si = G * sum rho * dV * z0 / (x^2 + y^2 + z0^2)^(3/2)
      gz_mgal = gz_si / 1e-5
    """
    assert rho_dhw.ndim == 3, f"rho_dhw should be (D,H,W), got {rho_dhw.shape}"
    D, H, W = rho_dhw.shape
    dx, dy, dz = sample
    device = rho_dhw.device
    rho = rho_dhw.to(dtype=dtype)

    dV = dx * dy * dz
    scale = G_SI * dV * SI_TO_MGAL  # multiply kernel by this to get mGal

    KH = 2 * H - 1
    KW = 2 * W - 1
    Hfull = H + KH - 1
    Wfull = W + KW - 1

    if pad_factor > 1:
        Hfft = int(pad_factor * Hfull)
        Wfft = int(pad_factor * Wfull)
    else:
        Hfft, Wfft = Hfull, Wfull

    rho_pad = torch.zeros((D, Hfft, Wfft), device=device, dtype=dtype)
    rho_pad[:, :H, :W] = rho

    R = torch.fft.rfft2(rho_pad, s=(Hfft, Wfft))

    p = (torch.arange(-(H - 1), H, device=device, dtype=dtype) + 0.5) * dx  # (KH,)
    q = (torch.arange(-(W - 1), W, device=device, dtype=dtype) + 0.5) * dy  # (KW,)
    px2 = p[:, None] ** 2
    qy2 = q[None, :] ** 2

    out = torch.zeros((len(heights), H, W), device=device, dtype=dtype)

    for oi, h_obs in enumerate(heights):
        acc_full = torch.zeros((Hfft, Wfft//2 + 1), device=device, dtype=torch.complex128)
        for k in range(D):
            z0 = dz * k + h_obs + (dz / 2.0)
            denom = (px2 + qy2 + (z0 ** 2)) ** 1.5
            K = scale * (z0 / denom)  # (KH,KW) already in mGal per (kg/m^3) voxel value

            Kpad = torch.zeros((Hfft, Wfft), device=device, dtype=dtype)
            Kpad[:KH, :KW] = K
            Kf = torch.fft.rfft2(Kpad, s=(Hfft, Wfft))
            acc_full += R[k] * Kf

        conv_full = torch.fft.irfft2(acc_full, s=(Hfft, Wfft))
        out[oi] = conv_full[(H - 1):(H - 1 + H), (W - 1):(W - 1 + W)]

    return out  # (Nh, H, W) in mGal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=str, default="", help="Optional density file (.npy/.npz). If omitted, random density is used.")
    ap.add_argument("--in_size", type=str, default="16,16,8", help="GraForward in_size as H,W,D (small demo default).")
    ap.add_argument("--heights", type=str, default="0,200,400", help="Heights list in meters, comma-separated.")
    ap.add_argument("--sample", type=str, default="100,100,100", help="Grid spacing dx,dy,dz (meters), comma-separated.")
    ap.add_argument("--weight", type=str, default=None,
                    help="Path to GraForward weight file (.pth). If missing, script forces load_from='' and will run set_params().")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32","float64"])
    ap.add_argument("--pad_factor", type=int, default=1, help="Extra FFT padding multiplier (>=1).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_npz", type=str, default="", help="Optional path to save outputs and diffs as npz.")
    ap.add_argument("--plot", action="store_true", help="Visualize comparison results.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    H, W, D = parse_ints(args.in_size)
    heights = parse_floats(args.heights)
    dx, dy, dz = parse_floats(args.sample)
    sample = (dx, dy, dz)
    out_size = (H, W, len(heights))

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    print(f'Device: {device}, dtype: {dtype}, in_size=(H,W,D)=({H},{W},{D}), heights={heights}, sample={sample}')

    # ----- load or create density (kg/m^3) -----
    if args.density:
        dens_np = load_density(args.density)
        dens_np = ensure_dhw(dens_np, D=D, H=H, W=W)
    else:
        # random demo density contrast (kg/m^3)
        dens_np = (100.0 * np.random.randn(D, H, W)).astype(np.float32)
    
    # nodes = cigvis.create_slices(dens_np)
    # cigvis.plot3D(nodes)

    rho = torch.from_numpy(dens_np).to(device=device, dtype=dtype)

    # ----- run GraForward (original) -----
    from gra_forward_ori import GraForward  # must be in same folder

    weight_path = args.weight
    if weight_path and os.path.exists(weight_path):
        load_from = weight_path
        print(f"[GraForward] Loading weights from: {weight_path}")
    else:
        load_from = ""
        print("[GraForward] Weight file not found. Will call set_params() (may be slow).")

    model = GraForward(in_size=(H, W, D), out_size=out_size, heights=heights, sample=list(sample), load_from=load_from)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    x = rho.unsqueeze(0).unsqueeze(0)          # (1,1,D,H,W)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    y_ori = model(x).squeeze(0).squeeze(0)     # (Nh,H,W) "internal scale" (missing G)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"GraForward execution time: {time.time() - t0:.4f} s")

    # Convert GraForward output -> mGal (kg/m^3, meters)
    # In GraForward: param = dV/1000 and missing G
    # Physical: multiply by G and use dV, so y_si = y_ori * (G * (dV / (dV/1000))) = y_ori * (G*1000)
    # Then mGal: y_mgal = y_si / 1e-5 = y_ori * (G*1000/1e-5) = y_ori * (G*1e8)
    scale_ori_to_mgal = G_SI * 1e8
    y_ori_mgal = y_ori * scale_ori_to_mgal

    # ----- run FFT physical forward (mGal) -----
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    y_fft_mgal = fft_forward_physical_mgal_like_graforward_geometry(
        rho, heights=heights, sample=sample, pad_factor=args.pad_factor, dtype=dtype
    )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"FFT forward execution time: {time.time() - t0:.4f} s")

    # ----- compare in mGal -----
    diff = y_fft_mgal - y_ori_mgal
    abs_max = diff.abs().max().item()
    abs_mean = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).sum().sqrt() / (y_ori_mgal.pow(2).sum().sqrt() + 1e-12)).item()

    print("\n=== Comparison metrics (physical mGal) ===")
    print(f"abs_max  = {abs_max:.6e} mGal")
    print(f"abs_mean = {abs_mean:.6e} mGal")
    print(f"rel_l2   = {rel_l2:.6e}")

    print("\nPer-height max|diff| (mGal):")
    for oi, h_obs in enumerate(heights):
        m = diff[oi].abs().max().item()
        print(f"  height={h_obs:g} m : {m:.6e}")

    if args.save_npz:
        out = {
            "rho_kgm3": rho.detach().cpu().numpy(),
            "y_ori_raw": y_ori.detach().cpu().numpy(),
            "y_ori_mgal": y_ori_mgal.detach().cpu().numpy(),
            "y_fft_mgal": y_fft_mgal.detach().cpu().numpy(),
            "diff_mgal": diff.detach().cpu().numpy(),
            "heights_m": np.array(heights, dtype=np.float32),
            "sample_m": np.array(sample, dtype=np.float32),
            "in_size": np.array([H, W, D], dtype=np.int32),
            "scale_ori_to_mgal": np.array([scale_ori_to_mgal], dtype=np.float64),
        }
        os.makedirs(os.path.dirname(args.save_npz) or ".", exist_ok=True)
        np.savez(args.save_npz, **out)
        print(f"\nSaved npz to: {args.save_npz}")

    if args.plot:
        import matplotlib.pyplot as plt
        
        y_ori_np = y_ori_mgal.detach().cpu().numpy()
        y_fft_np = y_fft_mgal.detach().cpu().numpy()
        diff_np = diff.detach().cpu().numpy()
        
        n_heights = len(heights)
        # Create a figure with N rows (heights) and 3 columns (GraForward, FFT, Diff)
        fig, axes = plt.subplots(n_heights, 3, figsize=(15, 4 * n_heights), squeeze=False)
        
        for i, h in enumerate(heights):
            # 1. GraForward (Original)
            im0 = axes[i, 0].imshow(y_ori_np[i], cmap='jet', origin='upper')
            axes[i, 0].set_title(f'GraForward (mGal) h={h}m')
            plt.colorbar(im0, ax=axes[i, 0])
            
            # 2. FFT (Physical)
            im1 = axes[i, 1].imshow(y_fft_np[i], cmap='jet', origin='upper')
            axes[i, 1].set_title(f'FFT (mGal) h={h}m')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # 3. Difference
            vmax_diff = max(abs(diff_np[i].min()), abs(diff_np[i].max()))
            if vmax_diff == 0: vmax_diff = 1e-9 # avoid warning for zero diff
            im2 = axes[i, 2].imshow(diff_np[i], cmap='seismic', origin='upper', vmin=-vmax_diff, vmax=vmax_diff)
            axes[i, 2].set_title(f'Diff (mGal) h={h}m\nMax err: {np.max(np.abs(diff_np[i])):.2e}')
            plt.colorbar(im2, ax=axes[i, 2])
        savepath = os.path.dirname(args.density)
        plt.savefig(os.path.join(savepath, "gravity_forward_comparison.png"), bbox_inches='tight', pad_inches=0.0)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    main()
