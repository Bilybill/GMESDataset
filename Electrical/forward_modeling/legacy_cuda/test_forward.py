import argparse
import time

import numpy as np
import torch

from mt_forward import MTForward3D, generate_mt_frequencies


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local MT 3D forward-modeling smoke test.")
    parser.add_argument(
        "--filepath",
        type=str,
        default="/home/wangyh/Project/GMESUni/MTForward3D/dome_density_1_resistivity.bin",
        help="Path to the resistivity .bin file.",
    )
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dx", type=float, default=50.0)
    parser.add_argument("--dy", type=float, default=50.0)
    parser.add_argument("--dz", type=float, default=50.0)
    parser.add_argument(
        "--freq-min",
        type=float,
        default=1,
        help="Optional minimum MT frequency in Hz. Must be paired with --freq-max.",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=1000,
        help="Optional maximum MT frequency in Hz. Must be paired with --freq-min.",
    )
    return parser.parse_args()


def resolve_frequency_list(freq_min: float | None, freq_max: float | None):
    if freq_min is None and freq_max is None:
        return None, "auto-selected frequencies"
    if freq_min is None or freq_max is None:
        raise ValueError("Please provide both --freq-min and --freq-max, or leave both unset for auto frequency mode.")
    if freq_min <= 0.0 or freq_max <= 0.0:
        raise ValueError("Frequency bounds must be positive.")
    freqs = generate_mt_frequencies(freq_min, freq_max)
    return freqs, f"user-selected frequencies in [{float(freq_min):g}, {float(freq_max):g}] Hz"


if __name__ == "__main__":
    args = parse_args()
    filepath = args.filepath
    nx, ny, nz = args.nx, args.ny, args.nz
    dx, dy, dz = args.dx, args.dy, args.dz

    print(f"Loading binary model from {filepath}")
    with open(filepath, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)

    if len(data) != nx * ny * nz:
        raise ValueError(f"Size mismatch: expected {nx * ny * nz}, got {len(data)}")

    rho_tensor = torch.from_numpy(data.astype(np.float64)).view(nz, ny, nx).permute(2, 1, 0).contiguous()

    freqs, freq_mode = resolve_frequency_list(args.freq_min, args.freq_max)

    print("Initializing PyTorch Operator...")
    operator = MTForward3D(freqs, dx, dy, dz)

    print(f"Running forward modeling with {freq_mode}...")
    start = time.time()
    app_res, phase = operator(rho_tensor)
    end = time.time()
    freqs = list(operator.last_freqs)

    print(f"\n✅ Forward modeling complete! Total Operator Time: {end - start:.2f} s")
    print(f"Selected Frequencies ({len(freqs)}): {freqs}")
    print(f"Apparent Resistivity Tensor Shape: {tuple(app_res.shape)}   (n_freqs, NX, NY, 2)")
    print(f"Phase Tensor Shape: {tuple(phase.shape)}")

    print(f"Example Target - Highest Freq ({freqs[0]} Hz), Center (X={nx // 2}, Y={ny // 2}):")
    print(f"  Apparent Resistivity (Zxy, Zyx):", app_res[0, nx // 2, ny // 2].numpy())
    print(f"  Phase (Zxy, Zyx):", phase[0, nx // 2, ny // 2].numpy())
