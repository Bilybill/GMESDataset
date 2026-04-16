from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - import is validated in the target training environment
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from experiments.datasets.modality_transforms import (
    format_mt_target,
    format_planar_target,
    format_seismic_target,
    log_standardize_volume,
    standardize_volume,
)


@dataclass(frozen=True)
class ForwardTaskSpec:
    name: str
    input_builder: Callable[[dict], np.ndarray]
    target_builder: Callable[[dict], np.ndarray | dict[str, np.ndarray]]
    required_status_key: str
    description: str


def _build_rho_input(bundle: dict) -> np.ndarray:
    return standardize_volume(bundle["rho_model"])[None, ...]


def _build_chi_input(bundle: dict) -> np.ndarray:
    return standardize_volume(bundle["chi_model"])[None, ...]


def _build_res_input(bundle: dict) -> np.ndarray:
    return log_standardize_volume(bundle["res_model"])[None, ...]


def _build_vp_input(bundle: dict) -> np.ndarray:
    return standardize_volume(bundle["vp_model"])[None, ...]


def _build_joint_input(bundle: dict) -> np.ndarray:
    return np.stack(
        [
            standardize_volume(bundle["vp_model"]),
            standardize_volume(bundle["rho_model"]),
            log_standardize_volume(bundle["res_model"]),
            standardize_volume(bundle["chi_model"]),
        ],
        axis=0,
    )


def _build_gravity_target(bundle: dict) -> np.ndarray:
    return format_planar_target(bundle["gravity_data"])


def _build_magnetic_target(bundle: dict) -> np.ndarray:
    return format_planar_target(bundle["magnetic_data"])


def _build_mt_target(bundle: dict) -> np.ndarray:
    return format_mt_target(bundle["mt_app_res"], bundle["mt_phase"], bundle.get("mt_freqs_hz"))


def _build_seismic_target(bundle: dict) -> np.ndarray:
    return format_seismic_target(bundle["seismic_data"])


def _build_joint_target(bundle: dict) -> dict[str, np.ndarray]:
    return {
        "gravity": _build_gravity_target(bundle),
        "magnetic": _build_magnetic_target(bundle),
        "mt": _build_mt_target(bundle),
        "seismic": _build_seismic_target(bundle),
    }


FORWARD_TASK_SPECS = {
    "rho_to_gravity": ForwardTaskSpec(
        name="rho_to_gravity",
        input_builder=_build_rho_input,
        target_builder=_build_gravity_target,
        required_status_key="gravity_status",
        description="Predict gravity observations from the density volume.",
    ),
    "chi_to_magnetic": ForwardTaskSpec(
        name="chi_to_magnetic",
        input_builder=_build_chi_input,
        target_builder=_build_magnetic_target,
        required_status_key="magnetic_status",
        description="Predict magnetic observations from the susceptibility volume.",
    ),
    "res_to_mt": ForwardTaskSpec(
        name="res_to_mt",
        input_builder=_build_res_input,
        target_builder=_build_mt_target,
        required_status_key="mt_status",
        description="Predict MT apparent resistivity and phase responses from the resistivity volume.",
    ),
    "vp_to_seismic": ForwardTaskSpec(
        name="vp_to_seismic",
        input_builder=_build_vp_input,
        target_builder=_build_seismic_target,
        required_status_key="seismic_status",
        description="Predict shot-gather observations from the velocity volume.",
    ),
    "joint_multiphysics": ForwardTaskSpec(
        name="joint_multiphysics",
        input_builder=_build_joint_input,
        target_builder=_build_joint_target,
        required_status_key="has_all_modalities",
        description="Jointly predict gravity, magnetic, MT, and seismic responses from aligned property volumes.",
    ),
}


def _to_tensor_tree(array_or_tree):
    if torch is None:
        return array_or_tree
    if isinstance(array_or_tree, dict):
        return {key: _to_tensor_tree(value) for key, value in array_or_tree.items()}
    return torch.from_numpy(array_or_tree)


def _clone_array_tree(array_or_tree):
    if isinstance(array_or_tree, dict):
        return {key: _clone_array_tree(value) for key, value in array_or_tree.items()}
    return array_or_tree.copy()


def _ones_mask_like(array_or_tree):
    if isinstance(array_or_tree, dict):
        return {key: _ones_mask_like(value) for key, value in array_or_tree.items()}
    return np.ones_like(array_or_tree, dtype=np.float32)


def _apply_target_scales_numpy(targets, target_scales):
    if target_scales is None:
        return _clone_array_tree(targets)
    if isinstance(targets, dict):
        return {
            key: _apply_target_scales_numpy(
                targets[key],
                target_scales.get(key) if isinstance(target_scales, dict) else None,
            )
            for key in targets
        }
    if isinstance(target_scales, dict):
        target_scales = target_scales.get("default")
    scale = float(target_scales) if target_scales not in (None, 0.0) else 1.0
    return (targets / scale).astype(np.float32, copy=False)


def infer_output_spec_from_sample(targets) -> dict[str, dict[str, tuple[int, ...] | int]]:
    if isinstance(targets, dict):
        return {
            name: {
                "out_channels": int(value.shape[0]),
                "output_shape": tuple(int(x) for x in value.shape[-2:]),
            }
            for name, value in targets.items()
        }
    return {
        "default": {
            "out_channels": int(targets.shape[0]),
            "output_shape": tuple(int(x) for x in targets.shape[-2:]),
        }
    }


class GMESForwardDataset(Dataset):
    def __init__(self, records: list[dict], task_name: str, target_scales=None):
        if task_name not in FORWARD_TASK_SPECS:
            raise ValueError(f"Unsupported forward task '{task_name}'. Available tasks: {sorted(FORWARD_TASK_SPECS)}")
        self.records = list(records)
        self.task = FORWARD_TASK_SPECS[task_name]
        self.target_scales = target_scales

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with np.load(record["bundle_path"], allow_pickle=True) as bundle:
            bundle_dict = {key: bundle[key] for key in bundle.files}
        input_array = self.task.input_builder(bundle_dict).astype(np.float32, copy=False)
        if self.task.name == "res_to_mt":
            target_array, target_mask = format_mt_target(
                bundle_dict["mt_app_res"],
                bundle_dict["mt_phase"],
                bundle_dict.get("mt_freqs_hz"),
                return_mask=True,
            )
            raw_targets = target_array.astype(np.float32, copy=False)
            target_masks = target_mask.astype(np.float32, copy=False)
        elif self.task.name == "joint_multiphysics":
            gravity = _build_gravity_target(bundle_dict).astype(np.float32, copy=False)
            magnetic = _build_magnetic_target(bundle_dict).astype(np.float32, copy=False)
            mt_target, mt_mask = format_mt_target(
                bundle_dict["mt_app_res"],
                bundle_dict["mt_phase"],
                bundle_dict.get("mt_freqs_hz"),
                return_mask=True,
            )
            seismic = _build_seismic_target(bundle_dict).astype(np.float32, copy=False)
            raw_targets = {
                "gravity": gravity,
                "magnetic": magnetic,
                "mt": mt_target.astype(np.float32, copy=False),
                "seismic": seismic,
            }
            target_masks = {
                "gravity": np.ones_like(gravity, dtype=np.float32),
                "magnetic": np.ones_like(magnetic, dtype=np.float32),
                "mt": mt_mask.astype(np.float32, copy=False),
                "seismic": np.ones_like(seismic, dtype=np.float32),
            }
        else:
            target_array_or_tree = self.task.target_builder(bundle_dict)
            if isinstance(target_array_or_tree, dict):
                raw_targets = {
                    key: value.astype(np.float32, copy=False)
                    for key, value in target_array_or_tree.items()
                }
            else:
                raw_targets = target_array_or_tree.astype(np.float32, copy=False)
            target_masks = _ones_mask_like(raw_targets)
        targets = _apply_target_scales_numpy(raw_targets, self.target_scales)

        item = {
            "inputs": torch.from_numpy(input_array) if torch is not None else input_array,
            "targets": _to_tensor_tree(targets),
            "raw_targets": _to_tensor_tree(raw_targets),
            "target_masks": _to_tensor_tree(target_masks),
            "metadata": {
                "bundle_path": record["bundle_path"],
                "partition": record["partition"],
                "background_id": record["background_id"],
                "anomaly_type": record["anomaly_type"],
                "task_name": self.task.name,
            },
        }
        return item


def infer_task_shapes(records: list[dict], task_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dataset = GMESForwardDataset(records[:1], task_name=task_name)
    sample = dataset[0]
    input_shape = tuple(sample["inputs"].shape)
    if isinstance(sample["targets"], dict):
        raise ValueError("infer_task_shapes only supports tensor targets. Use infer_output_spec_from_sample for joint tasks.")
    target_shape = tuple(sample["targets"].shape)
    return input_shape, target_shape
