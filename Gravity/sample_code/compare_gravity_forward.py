#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare gravity forward outputs between:
  (1) GraForward (matrix/Linear) implementation from gra_forward_ori.py
  (2) An FFT-based implementation that reproduces the SAME discrete kernel used in GraForward.set_params()

Why this script exists
- GraForward is "direct summation baked into a giant linear layer" (y = W x).
- FFT version computes the same mapping via linear convolution in x-y per depth slice, then sums over depth.

Important notes
- If you do NOT have a precomputed weight file (accurate.pth), GraForward will call set_params(),
  which is O(H*W*H*W*D*Nh) and becomes very slow for large grids. Use small sizes for demo.
- This script compares *numerical equality* under the discrete kernel used by GraForward, NOT physical scaling with G.
"""

import os
import argparse
import numpy as np
import torch

def parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]

def parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip() != ""]

def load_density(path: str):
    """
    Load density model from .npy or .npz.

    Returns numpy array.
    Supported shapes:
      - (D, H, W)  [preferred]
      - (H, W, D)  [will be transposed to (D, H, W)]
    """
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        # common keys: density, rho, model, data
        for k in ["density", "rho", "model", "data"]:
            if k in z:
                arr = z[k]
                break
        else:
            # fallback: first key
            arr = z[list(z.keys())[0]]
    else:
        raise ValueError(f"Unsupported density file type: {path}")

    if arr.ndim != 3:
        raise ValueError(f"Density must be 3D. Got shape={arr.shape}")

    return arr

def ensure_dhw(arr: np.ndarray, D: int, H: int, W: int):
    """
    Ensure density array is (D,H,W).
    """
    if arr.shape == (D, H, W):
        return arr
    if arr.shape == (H, W, D):
        return np.transpose(arr, (2, 0, 1))
    # try other permutations
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    for p in perms:
        if tuple(arr.shape[i] for i in p) == (D,H,W):
            return np.transpose(arr, p)
    raise ValueError(f"Cannot reshape/transpose density to (D,H,W). density.shape={arr.shape}, target={(D,H,W)}")

@torch.no_grad()
def fft_forward_like_graforward(
    rho_dhw: torch.Tensor,
    heights: list[float],
    sample: tuple[float,float,float],
    pad_factor: int = 1,
    dtype: torch.dtype = torch.float64,
):
    """
    Reproduce GraForward.set_params() kernel, but computed via FFT convolution.

    GraForward kernel:
      weight[o,m,n,k,i,j] = z0 * param / ((sh*i - sh*m - sh/2)^2 + (sw*j - sw*n - sw/2)^2 + z0^2)^(3/2)
      z0 = sd*k + heights[o] + sd/2
      param = sh*sw*sd / 1000

    We rewrite in terms of offsets p = m - i, q = n - j (standard convolution):
      (sh*i - sh*m - sh/2)^2 = sh^2 * (p + 0.5)^2
      (sw*j - sw*n - sw/2)^2 = sw^2 * (q + 0.5)^2

    Output:
      gz[o, m, n] = sum_k sum_{i,j} rho[k,i,j] * K_{k,o}(p=m-i, q=n-j)
    """
    assert rho_dhw.ndim == 3, f"rho_dhw should be (D,H,W), got {rho_dhw.shape}"
    D, H, W = rho_dhw.shape
    sh, sw, sd = sample
    device = rho_dhw.device
    rho = rho_dhw.to(dtype=dtype)

    # "param" exactly as in GraForward
    param = (sh * sw * sd) / 1000.0

    # kernel size to reproduce "same" output for HxW grid with offsets in [-(H-1), +(H-1)]
    KH = 2 * H - 1
    KW = 2 * W - 1
    Hfull = H + KH - 1  # = 3H - 2
    Wfull = W + KW - 1  # = 3W - 2

    # Optional extra padding for speed/accuracy (still linear conv after cropping)
    if pad_factor > 1:
        Hfft = int(pad_factor * Hfull)
        Wfft = int(pad_factor * Wfull)
    else:
        Hfft, Wfft = Hfull, Wfull

    # Prepare padded rho once per depth slice
    # rho_pad: (D, Hfft, Wfft) with rho in top-left HxW
    rho_pad = torch.zeros((D, Hfft, Wfft), device=device, dtype=dtype)
    rho_pad[:, :H, :W] = rho

    # FFT of rho slices (batched)
    R = torch.fft.rfft2(rho_pad, s=(Hfft, Wfft))

    # Offsets p,q for kernel indices (standard conv): p=m-i in [-(H-1), H-1], q=n-j in [-(W-1), W-1]
    p = (torch.arange(-(H - 1), H, device=device, dtype=dtype) + 0.5) * sh  # (KH,)
    q = (torch.arange(-(W - 1), W, device=device, dtype=dtype) + 0.5) * sw  # (KW,)
    px2 = p[:, None] ** 2  # (KH,1)
    qy2 = q[None, :] ** 2  # (1,KW)

    out = torch.zeros((len(heights), H, W), device=device, dtype=dtype)

    for oi, h_obs in enumerate(heights):
        acc_full = torch.zeros((Hfft, Wfft//2 + 1), device=device, dtype=torch.complex128)
        # sum over depth
        for k in range(D):
            z0 = sd * k + h_obs + (sd / 2.0)
            denom = (px2 + qy2 + (z0 ** 2)) ** 1.5  # (KH,KW)
            K = (z0 * param) / denom  # (KH,KW)

            # pad kernel to FFT size (top-left placement = standard linear convolution indexing)
            Kpad = torch.zeros((Hfft, Wfft), device=device, dtype=dtype)
            Kpad[:KH, :KW] = K
            Kf = torch.fft.rfft2(Kpad, s=(Hfft, Wfft))

            acc_full += R[k] * Kf

        conv_full = torch.fft.irfft2(acc_full, s=(Hfft, Wfft))  # (Hfft,Wfft), real

        # Crop "same" output: indices [H-1:H-1+H, W-1:W-1+W]
        out[oi] = conv_full[(H - 1):(H - 1 + H), (W - 1):(W - 1 + W)]

    return out  # (Nh, H, W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=str, default="", help="Optional density file (.npy/.npz). If omitted, random density is used.")
    ap.add_argument("--in_size", type=str, default="16,16,8", help="GraForward in_size as H,W,D (small demo default).")
    ap.add_argument("--heights", type=str, default="0,200,400", help="Heights list in meters, comma-separated.")
    ap.add_argument("--sample", type=str, default="100,100,100", help="Sample spacing sh,sw,sd (meters), comma-separated.")
    ap.add_argument("--weight", type=str, default="work_dirs/gra_forward/accurate.pth",
                    help="Path to GraForward weight file (.pth). If missing, script forces load_from='' and will run set_params().")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32","float64"])
    ap.add_argument("--pad_factor", type=int, default=1, help="Extra FFT padding multiplier (>=1).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_npz", type=str, default="", help="Optional path to save outputs and diffs as npz.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    H, W, D = parse_ints(args.in_size)
    heights = parse_floats(args.heights)
    sh, sw, sd = parse_floats(args.sample)
    sample = (sh, sw, sd)
    out_size = (H, W, len(heights))

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    # ----- load or create density -----
    if args.density:
        dens_np = load_density(args.density)
        dens_np = ensure_dhw(dens_np, D=D, H=H, W=W)
    else:
        # random demo density (keep scale moderate)
        dens_np = np.random.randn(D, H, W).astype(np.float32)

    rho = torch.from_numpy(dens_np).to(device=device, dtype=dtype)

    # ----- run GraForward (original) -----
    # Import from local file
    from gra_forward_ori import GraForward  # file should be in same folder as this script

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

    x = rho.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    y_ori = model(x).squeeze(0).squeeze(0)  # (Nh,H,W)

    # ----- run FFT forward (reproducing GraForward kernel) -----
    y_fft = fft_forward_like_graforward(rho, heights=heights, sample=sample, pad_factor=args.pad_factor, dtype=dtype)

    # ----- compare -----
    diff = y_fft - y_ori
    abs_max = diff.abs().max().item()
    abs_mean = diff.abs().mean().item()
    rel_l2 = (diff.pow(2).sum().sqrt() / (y_ori.pow(2).sum().sqrt() + 1e-12)).item()

    print("\n=== Comparison metrics (FFT vs GraForward) ===")
    print(f"abs_max  = {abs_max:.6e}")
    print(f"abs_mean = {abs_mean:.6e}")
    print(f"rel_l2   = {rel_l2:.6e}")

    # per height
    print("\nPer-height max|diff|:")
    for oi, h_obs in enumerate(heights):
        m = diff[oi].abs().max().item()
        print(f"  height={h_obs:g} m : {m:.6e}")

    # optional save
    if args.save_npz:
        out = {
            "rho": rho.detach().cpu().numpy(),
            "y_ori": y_ori.detach().cpu().numpy(),
            "y_fft": y_fft.detach().cpu().numpy(),
            "diff": diff.detach().cpu().numpy(),
            "heights": np.array(heights, dtype=np.float32),
            "sample": np.array(sample, dtype=np.float32),
            "in_size": np.array([H,W,D], dtype=np.int32),
        }
        os.makedirs(os.path.dirname(args.save_npz) or ".", exist_ok=True)
        np.savez(args.save_npz, **out)
        print(f"\nSaved npz to: {args.save_npz}")

if __name__ == "__main__":
    main()
