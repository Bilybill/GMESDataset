import torch
from deepwave import scalar
import argparse
import os
import numpy as np

from utils_unified import load_config, load_velocity_model, get_wavelet, setup_acquisition


def main():
    parser = argparse.ArgumentParser(description="Seismic Forward Modeling using Deepwave (Unified Acquisition)")
    parser.add_argument("--config", type=str, default="config_unified.yaml", help="Path to configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_config(args.config)

    # Device
    dev = config.get("simulation", {}).get("device", "cuda")
    device = torch.device("cuda" if (dev == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Velocity
    v = load_velocity_model(config).to(device)

    # Grid spacing
    mode = config["simulation"]["mode"]
    dx_val = float(config["model"]["dx"])
    dz_val = float(config["model"]["dz"])
    if mode == "2D":
        dx = [dx_val, dz_val]
        print(f"Grid spacing: dx={dx_val}, dz={dz_val}")
    else:
        dy_val = float(config["model"]["dy"])
        dx = [dx_val, dy_val, dz_val]
        print(f"Grid spacing: dx={dx_val}, dy={dy_val}, dz={dz_val}")

    # Time
    dt = float(config["time"]["dt"])

    # Wavelet
    wavelet = get_wavelet(config, device)

    # Acquisition geometry (IMPORTANT: before source_amplitudes)
    source_locations, receiver_locations = setup_acquisition(config, device)

    n_shots = int(source_locations.shape[0])
    n_src = int(source_locations.shape[1])

    # Source amplitudes: [n_shots, n_src, nt]
    source_amplitudes = wavelet.unsqueeze(0).unsqueeze(0).repeat(n_shots, n_src, 1)

    print("Starting Propagator...")
    print(f"Model shape: {tuple(v.shape)}")
    print(f"Shots: {n_shots}")
    print(f"Receivers/shot: {receiver_locations.shape[1]}")

    pml_freq = float(config["source"].get("freq", 25.0))

    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=pml_freq,
        accuracy=int(config.get("simulation", {}).get("accuracy", 4)),
    )

    receiver_data = out[-1]
    print("Simulation Complete.")
    print(f"Output shape: {tuple(receiver_data.shape)}")

    save_path = config.get("output", {}).get("save_path", "shot_gathers.npy")
    save_dir = os.path.dirname(save_path)
    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)

    np.save(save_path, receiver_data.detach().cpu().numpy())
    print(f"Saved data to {save_path}")


if __name__ == "__main__":
    main()
