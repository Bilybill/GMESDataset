from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from experiments.datasets.gmes_inverse_dataset import AVAILABLE_MODALITIES, MODALITY_STATUS_KEYS, normalize_modalities
from experiments.datasets.modality_transforms import (
    format_mt_target,
    format_planar_input,
    format_seismic_target,
    log_standardize_volume,
    standardize_volume,
)


TARGET_FIELDS = ("vp", "rho", "res", "chi")


def filter_records_for_joint_inversion(records: list[dict], modalities: Iterable[str]) -> list[dict]:
    selected = normalize_modalities(modalities)
    return [
        record
        for record in records
        if all(record.get(MODALITY_STATUS_KEYS[modality]) == "ok" for modality in selected)
    ]


def _to_tensor(array: np.ndarray):
    if torch is None:
        return array
    return torch.from_numpy(array)


def _downsample_volume_average(volume: np.ndarray, output_shape: tuple[int, int, int]) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    if tuple(volume.shape) == tuple(output_shape):
        return volume.astype(np.float32, copy=False)

    target_z, target_y, target_x = output_shape
    if (
        volume.shape[0] % target_z == 0
        and volume.shape[1] % target_y == 0
        and volume.shape[2] % target_x == 0
    ):
        factor_z = volume.shape[0] // target_z
        factor_y = volume.shape[1] // target_y
        factor_x = volume.shape[2] // target_x
        return volume.reshape(target_z, factor_z, target_y, factor_y, target_x, factor_x).mean(axis=(1, 3, 5)).astype(np.float32, copy=False)

    z_bins = np.array_split(np.arange(volume.shape[0]), target_z)
    y_bins = np.array_split(np.arange(volume.shape[1]), target_y)
    x_bins = np.array_split(np.arange(volume.shape[2]), target_x)
    reduced = np.zeros(output_shape, dtype=np.float32)
    for iz, z_idx in enumerate(z_bins):
        for iy, y_idx in enumerate(y_bins):
            for ix, x_idx in enumerate(x_bins):
                reduced[iz, iy, ix] = float(volume[np.ix_(z_idx, y_idx, x_idx)].mean())
    return reduced


def _build_modality_array(bundle: dict, modality: str) -> np.ndarray:
    if modality == "gravity":
        return format_planar_input(bundle["gravity_data"]).astype(np.float32, copy=False)
    if modality == "magnetic":
        return format_planar_input(bundle["magnetic_data"]).astype(np.float32, copy=False)
    if modality == "mt":
        return format_mt_target(bundle["mt_app_res"], bundle["mt_phase"], bundle.get("mt_freqs_hz")).astype(np.float32, copy=False)
    if modality == "seismic":
        return format_seismic_target(bundle["seismic_data"]).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported modality '{modality}'.")


def _build_target_volume(bundle: dict, output_shape: tuple[int, int, int]) -> np.ndarray:
    targets = [
        standardize_volume(_downsample_volume_average(bundle["vp_model"], output_shape)),
        standardize_volume(_downsample_volume_average(bundle["rho_model"], output_shape)),
        log_standardize_volume(_downsample_volume_average(bundle["res_model"], output_shape)),
        standardize_volume(_downsample_volume_average(bundle["chi_model"], output_shape)),
    ]
    return np.stack(targets, axis=0).astype(np.float32, copy=False)


def _required_keys(modalities: tuple[str, ...]) -> list[str]:
    keys = {"vp_model", "rho_model", "res_model", "chi_model"}
    for modality in modalities:
        if modality == "gravity":
            keys.add("gravity_data")
        elif modality == "magnetic":
            keys.add("magnetic_data")
        elif modality == "mt":
            keys.update(("mt_app_res", "mt_phase", "mt_freqs_hz"))
        elif modality == "seismic":
            keys.add("seismic_data")
    return sorted(keys)


class GMESJointInversionDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        modalities: Iterable[str] = AVAILABLE_MODALITIES,
        target_shape: tuple[int, int, int] = (64, 64, 64),
    ):
        self.records = list(records)
        self.modalities = normalize_modalities(modalities)
        self.target_shape = tuple(int(x) for x in target_shape)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        bundle_path = record["bundle_path"]
        try:
            with np.load(bundle_path, allow_pickle=True) as bundle:
                bundle_dict = {}
                for key in _required_keys(self.modalities):
                    if key not in bundle.files:
                        if key == "mt_freqs_hz":
                            continue
                        raise KeyError(f"missing key '{key}' in {bundle_path}")
                    bundle_dict[key] = bundle[key]
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(f"failed loading joint-inversion sample: {bundle_path}: {exc}") from exc

        modalities = {
            modality: _to_tensor(_build_modality_array(bundle_dict, modality))
            for modality in self.modalities
        }
        target = _build_target_volume(bundle_dict, self.target_shape)
        return {
            "modalities": modalities,
            "target": _to_tensor(target),
            "metadata": {
                "bundle_path": bundle_path,
                "partition": record["partition"],
                "background_id": record["background_id"],
                "anomaly_type": record["anomaly_type"],
            },
        }
