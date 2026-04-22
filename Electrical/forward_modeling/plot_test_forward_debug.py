import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_NPZ = SCRIPT_DIR / "debug_outputs" / "plot_fig1_data_mt_forward_debug.npz"
OUTPUT_DIR = SCRIPT_DIR / "debug_outputs" / "frequency_axis_slices_from_saved"


def percentile_clim(values, lower=5.0, upper=95.0):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return [0.0, 1.0]

    vmin = float(np.percentile(finite, lower))
    vmax = float(np.percentile(finite, upper))
    if vmax <= vmin:
        vmax = vmin + 1.0e-6
    return [vmin, vmax]


def mapped_z_index(freq_index, n_freq, nz):
    if n_freq <= 1:
        return nz // 2
    return int(round(freq_index * (nz - 1) / (n_freq - 1)))


def save_frequency_slice_figure(
    ifreq,
    freq_hz,
    app_res,
    phase,
    res_model,
    app_clim,
    phase_clim,
    res_clim,
    output_dir,
):
    n_freq = app_res.shape[0]
    nz = res_model.shape[2]
    z_idx = mapped_z_index(ifreq, n_freq, nz)

    log_app = np.log10(np.clip(app_res, 1.0e-6, None))
    log_res = np.log10(np.clip(res_model, 1.0e-6, None))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    panels = [
        (
            axes[0, 0],
            log_app[ifreq, :, :, 0].T,
            "TE log10 apparent resistivity",
            "log10(Ohm-m)",
            app_clim,
        ),
        (
            axes[0, 1],
            log_app[ifreq, :, :, 1].T,
            "TM log10 apparent resistivity",
            "log10(Ohm-m)",
            app_clim,
        ),
        (
            axes[0, 2],
            log_res[:, :, z_idx].T,
            f"Resistivity model Z slice {z_idx}",
            "log10(Ohm-m)",
            res_clim,
        ),
        (
            axes[1, 0],
            phase[ifreq, :, :, 0].T,
            "TE phase",
            "degree",
            phase_clim,
        ),
        (
            axes[1, 1],
            phase[ifreq, :, :, 1].T,
            "TM phase",
            "degree",
            phase_clim,
        ),
    ]

    for ax, image, title, cbar_label, clim in panels:
        im = ax.imshow(
            image,
            origin="lower",
            cmap="jet",
            aspect="equal",
        )
        ax.set_title(f"{title}\nf={float(freq_hz):.6g} Hz")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        fig.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        "MT forward debug\n"
        f"frequency index: {ifreq}/{n_freq - 1}\n"
        f"frequency: {float(freq_hz):.6g} Hz\n"
        f"mapped model Z slice: {z_idx}/{nz - 1}\n"
        f"app_res shape: {app_res.shape}\n"
        f"res_model shape: {res_model.shape}",
        ha="left",
        va="top",
        fontsize=12,
    )

    output_path = output_dir / f"freq_{ifreq:03d}_{float(freq_hz):.6g}Hz.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(DEBUG_NPZ, allow_pickle=True) as data:
        res_model = data["res_model_downsampled"].astype(np.float32)
        app_res = data["app_res"].astype(np.float32)
        phase = data["phase"].astype(np.float32)
        freqs_hz = data["freqs_hz"].astype(np.float32)

    log_app = np.log10(np.clip(app_res, 1.0e-6, None))
    log_res = np.log10(np.clip(res_model, 1.0e-6, None))

    app_clim = percentile_clim(log_app, lower=5.0, upper=95.0)
    phase_clim = percentile_clim(phase, lower=5.0, upper=95.0)
    res_clim = percentile_clim(log_res, lower=5.0, upper=95.0)

    for ifreq, freq_hz in enumerate(freqs_hz):
        save_frequency_slice_figure(
            ifreq=ifreq,
            freq_hz=freq_hz,
            app_res=app_res,
            phase=phase,
            res_model=res_model,
            app_clim=app_clim,
            phase_clim=phase_clim,
            res_clim=res_clim,
            output_dir=OUTPUT_DIR,
        )

    print(f"Saved frequency-axis debug plots to: {OUTPUT_DIR}")
