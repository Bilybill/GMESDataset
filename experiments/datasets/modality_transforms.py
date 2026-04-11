from __future__ import annotations

import numpy as np


def standardize_volume(volume: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    mean = float(volume.mean())
    std = float(volume.std())
    if std < eps:
        std = 1.0
    return (volume - mean) / std


def log_standardize_volume(volume: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    safe = np.clip(volume, eps, None)
    return standardize_volume(np.log10(safe), eps=eps)


def standardize_map(array: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    return standardize_volume(np.asarray(array, dtype=np.float32), eps=eps)


def format_planar_target(array_2d: np.ndarray) -> np.ndarray:
    array_2d = np.asarray(array_2d, dtype=np.float32)
    if array_2d.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array_2d.shape}.")
    return array_2d[None, ...]


def format_planar_input(array_2d: np.ndarray) -> np.ndarray:
    return format_planar_target(standardize_map(array_2d))


def format_mt_target(mt_app_res: np.ndarray, mt_phase: np.ndarray, app_res_eps: float = 1.0e-6) -> np.ndarray:
    app_res = np.asarray(mt_app_res, dtype=np.float32)
    phase = np.asarray(mt_phase, dtype=np.float32)
    if app_res.shape != phase.shape:
        raise ValueError(f"MT apparent-resistivity shape {app_res.shape} does not match phase shape {phase.shape}.")
    if app_res.ndim != 4:
        raise ValueError(f"Expected MT tensors with shape (freq, x, y, pol), got {app_res.shape}.")
    app_res = np.log10(np.clip(app_res, app_res_eps, None))
    phase = phase / 180.0
    # (freq, x, y, pol) -> (channels, x, y)
    app_res = np.transpose(app_res, (0, 3, 1, 2)).reshape(-1, app_res.shape[1], app_res.shape[2])
    phase = np.transpose(phase, (0, 3, 1, 2)).reshape(-1, phase.shape[1], phase.shape[2])
    return np.concatenate([app_res, phase], axis=0)


def format_seismic_target(
    seismic_data: np.ndarray,
    shot_stride: int = 1,
    receiver_stride: int = 16,
    time_stride: int = 4,
    amplitude_clip_quantile: float | None = 0.995,
) -> np.ndarray:
    seismic = np.asarray(seismic_data, dtype=np.float32)
    if seismic.ndim != 3:
        raise ValueError(f"Expected seismic data with shape (shot, receiver, time), got {seismic.shape}.")

    seismic = seismic[:: max(int(shot_stride), 1), :: max(int(receiver_stride), 1), :: max(int(time_stride), 1)]
    if amplitude_clip_quantile is not None:
        quantile = float(np.quantile(np.abs(seismic), amplitude_clip_quantile))
        if quantile > 0.0:
            seismic = np.clip(seismic, -quantile, quantile) / quantile
    return seismic


def downsample_binary_mask(mask: np.ndarray, output_shape: tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    mask = np.asarray(mask > 0, dtype=np.float32)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}.")
    target_z, target_y, target_x = output_shape

    if (
        mask.shape[0] % target_z == 0
        and mask.shape[1] % target_y == 0
        and mask.shape[2] % target_x == 0
    ):
        factor_z = mask.shape[0] // target_z
        factor_y = mask.shape[1] // target_y
        factor_x = mask.shape[2] // target_x
        reduced = mask.reshape(target_z, factor_z, target_y, factor_y, target_x, factor_x).mean(axis=(1, 3, 5))
        return (reduced >= 0.5).astype(np.float32, copy=False)

    z_bins = np.array_split(np.arange(mask.shape[0]), target_z)
    y_bins = np.array_split(np.arange(mask.shape[1]), target_y)
    x_bins = np.array_split(np.arange(mask.shape[2]), target_x)

    reduced = np.zeros(output_shape, dtype=np.float32)
    for iz, z_idx in enumerate(z_bins):
        for iy, y_idx in enumerate(y_bins):
            for ix, x_idx in enumerate(x_bins):
                block = mask[np.ix_(z_idx, y_idx, x_idx)]
                reduced[iz, iy, ix] = float(block.mean() >= 0.5)
    return reduced
