import argparse
import os
import sys
import time
from pathlib import Path

# GPU_INDEX = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
FORWARD_MODELING_DIR = SCRIPT_DIR.parent
GMES_ROOT = SCRIPT_DIR.parents[2]

if str(FORWARD_MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(FORWARD_MODELING_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt

from mt_forward import MTForward3D, generate_mt_frequencies


DEFAULT_MODEL_BUNDLE = (
    GMES_ROOT
    / "DATAFOLDER"
    / "PretrainDataset"
    / "train-river"
    / "braided"
    / "AYL-00018"
    / "salt_dome"
    / "model_bundle.npz"
)
DEFAULT_OUTPUT_DIR = FORWARD_MODELING_DIR / "debug_outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug MT forward modeling on the resistivity model used by plot_fig1_data.py."
    )
    parser.add_argument(
        "--model-bundle",
        type=str,
        default=str(DEFAULT_MODEL_BUNDLE),
        help="Path to model_bundle.npz containing res_model.",
    )
    parser.add_argument("--target-nx", type=int, default=50)
    parser.add_argument("--target-ny", type=int, default=50)
    parser.add_argument("--target-nz", type=int, default=50)
    parser.add_argument(
        "--freq-min",
        type=float,
        default=0.01,
        help="Minimum MT frequency in Hz.",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=1000.0,
        help="Maximum MT frequency in Hz.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for debug outputs.",
    )
    parser.add_argument(
        "--mfem-device",
        type=str,
        default=os.environ.get("GMES_MT_MFEM_DEVICE", "cpu"),
        help='MFEM device string, for example "cpu" or "cuda".',
    )
    parser.add_argument(
        "--partial-assembly",
        action="store_true",
        help="Enable MFEM partial assembly / matrix-free preconditioning path.",
    )
    return parser.parse_args()


def scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr


def downsample_res_model(res_model, target_shape):
    """Match the project MT preprocessing: interpolate in conductivity domain."""
    res_tensor = torch.as_tensor(res_model, dtype=torch.float32).contiguous()
    sigma_tensor = torch.clamp(res_tensor, min=1.0e-3).reciprocal()
    sigma_5d = sigma_tensor.unsqueeze(0).unsqueeze(0)
    sigma_down = torch.nn.functional.interpolate(
        sigma_5d,
        size=target_shape,
        mode="trilinear",
        align_corners=True,
    ).squeeze().contiguous()
    return torch.clamp(sigma_down, min=1.0e-6).reciprocal()


def print_stats(name, arr):
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if finite.size == 0:
        print(f"  finite values: none")
        return
    print(
        "  min/max/median/p95 = "
        f"{finite.min():.6g} / {finite.max():.6g} / "
        f"{np.median(finite):.6g} / {np.percentile(finite, 95.0):.6g}"
    )
    print(f"  nan count={np.isnan(arr).sum()}, inf count={np.isinf(arr).sum()}")


def percentile_clim(values, lower=5.0, upper=95.0):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return [0.0, 1.0]
    vmin = float(np.percentile(finite, lower))
    vmax = float(np.percentile(finite, upper))
    if vmax <= vmin:
        vmax = vmin + 1.0e-6
    return [vmin, vmax]


def plot_frequency_axis_debug(app_res, phase, freqs, res_model, output_dir):
    plot_dir = Path(output_dir) / "frequency_axis_slices"
    plot_dir.mkdir(parents=True, exist_ok=True)

    log_app = np.log10(np.clip(app_res, 1.0e-6, None))
    log_res = np.log10(np.clip(res_model, 1.0e-6, None))

    app_clim = percentile_clim(log_app, lower=5.0, upper=95.0)
    phase_clim = percentile_clim(phase, lower=5.0, upper=95.0)
    res_clim = percentile_clim(log_res, lower=5.0, upper=95.0)

    n_freq = int(app_res.shape[0])
    nz = int(res_model.shape[2])

    for ifreq, freq in enumerate(freqs):
        if n_freq == 1:
            z_idx = nz // 2
        else:
            z_idx = int(round(ifreq * (nz - 1) / (n_freq - 1)))

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

        panels = [
            (
                axes[0, 0],
                log_app[ifreq, :, :, 0].T,
                "jet",
                app_clim,
                f"TE log10 apparent resistivity\nf={float(freq):.6g} Hz",
                "log10(Ohm-m)",
            ),
            (
                axes[0, 1],
                log_app[ifreq, :, :, 1].T,
                "jet",
                app_clim,
                f"TM log10 apparent resistivity\nf={float(freq):.6g} Hz",
                "log10(Ohm-m)",
            ),
            (
                axes[0, 2],
                log_res[:, :, z_idx].T,
                "jet",
                res_clim,
                f"Downsampled resistivity model\nZ slice={z_idx}",
                "log10(Ohm-m)",
            ),
            (
                axes[1, 0],
                phase[ifreq, :, :, 0].T,
                "jet",
                phase_clim,
                f"TE phase\nf={float(freq):.6g} Hz",
                "degree",
            ),
            (
                axes[1, 1],
                phase[ifreq, :, :, 1].T,
                "jet",
                phase_clim,
                f"TM phase\nf={float(freq):.6g} Hz",
                "degree",
            ),
        ]

        for ax, img, cmap, clim, title, cbar_label in panels:
            im = ax.imshow(
                img,
                origin="lower",
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                aspect="equal",
            )
            ax.set_title(title)
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")
            fig.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)

        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.02,
            0.95,
            "Frequency-axis debug\n"
            f"freq index = {ifreq}/{n_freq - 1}\n"
            f"freq = {float(freq):.6g} Hz\n"
            f"mapped Z slice = {z_idx}/{nz - 1}",
            ha="left",
            va="top",
            fontsize=12,
        )

        fig.savefig(plot_dir / f"freq_{ifreq:03d}_{float(freq):.6g}Hz.png", dpi=180)
        plt.close(fig)

    print(f"Saved frequency-axis debug plots: {plot_dir}")


if __name__ == "__main__":
    args = parse_args()
    model_bundle = Path(args.model_bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_shape = (args.target_nx, args.target_ny, args.target_nz)

    # print(f"Using physical GPU index: {GPU_INDEX} via CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Loading model bundle: {model_bundle}")
    with np.load(model_bundle, allow_pickle=True) as data:
        res_model = data["res_model"].astype(np.float32)
        dx = float(scalar(data["dx"]))
        dy = float(scalar(data["dy"]))
        dz = float(scalar(data["dz"]))

    nx, ny, nz = res_model.shape
    mt_dx = dx * nx / target_shape[0]
    mt_dy = dy * ny / target_shape[1]
    mt_dz = dz * nz / target_shape[2]

    print_stats("Original resistivity model", res_model)
    print(f"Original spacing: dx={dx:g}, dy={dy:g}, dz={dz:g}")

    res_down = downsample_res_model(res_model, target_shape)
    res_down_np = res_down.detach().cpu().numpy().astype(np.float32)
    print_stats("Downsampled resistivity model", res_down_np)
    print(f"MT spacing after downsampling: dx={mt_dx:g}, dy={mt_dy:g}, dz={mt_dz:g}")

    downsampled_bin = (
        output_dir
        / f"plot_fig1_data_res_model_{args.target_nx}x{args.target_ny}x{args.target_nz}.bin"
    )
    res_down_np.astype(np.float32).tofile(downsampled_bin)
    print(f"Saved downsampled resistivity binary: {downsampled_bin}")

    freqs = generate_mt_frequencies(args.freq_min, args.freq_max)
    print(f"Running MT forward with {len(freqs)} frequencies from {args.freq_min:g} to {args.freq_max:g} Hz")
    print(f"Frequencies: {freqs}")

    rho_tensor = res_down.to(torch.float64).contiguous()
    print(f"MFEM device request: {args.mfem_device}")
    print(f"Use partial assembly: {args.partial_assembly}")
    operator = MTForward3D(
        freqs,
        mt_dx,
        mt_dy,
        mt_dz,
        device=args.mfem_device,
        use_partial_assembly=args.partial_assembly,
    )

    start = time.time()
    app_res, phase = operator(rho_tensor)
    elapsed = time.time() - start

    app_res_np = app_res.detach().cpu().numpy().astype(np.float32)
    phase_np = phase.detach().cpu().numpy().astype(np.float32)
    freqs_np = np.asarray(operator.last_freqs, dtype=np.float32)

    print(f"\nForward modeling complete in {elapsed:.2f} s")
    print_stats("Apparent resistivity", app_res_np)
    print_stats("Phase", phase_np)

    cx, cy = target_shape[0] // 2, target_shape[1] // 2
    print(f"Center station index: X={cx}, Y={cy}")
    print(f"Highest frequency {freqs_np[0]:g} Hz app_res xy/yx: {app_res_np[0, cx, cy]}")
    print(f"Highest frequency {freqs_np[0]:g} Hz phase xy/yx: {phase_np[0, cx, cy]}")
    print(f"Lowest frequency {freqs_np[-1]:g} Hz app_res xy/yx: {app_res_np[-1, cx, cy]}")
    print(f"Lowest frequency {freqs_np[-1]:g} Hz phase xy/yx: {phase_np[-1, cx, cy]}")

    output_npz = output_dir / "plot_fig1_data_mt_forward_debug.npz"
    np.savez_compressed(
        output_npz,
        res_model_downsampled=res_down_np,
        mt_dx=np.array(mt_dx, dtype=np.float32),
        mt_dy=np.array(mt_dy, dtype=np.float32),
        mt_dz=np.array(mt_dz, dtype=np.float32),
        freqs_hz=freqs_np,
        app_res=app_res_np,
        phase=phase_np,
        source_model_bundle=np.array(str(model_bundle)),
    )
    print(f"Saved debug forward results: {output_npz}")

    plot_frequency_axis_debug(
        app_res_np,
        phase_np,
        freqs_np,
        res_down_np,
        output_dir,
    )
