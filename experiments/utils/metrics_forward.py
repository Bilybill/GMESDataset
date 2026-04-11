from __future__ import annotations

import math


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def relative_l2(prediction, target, eps: float = 1.0e-8):
    numerator = (prediction - target).pow(2).sum(dim=tuple(range(1, prediction.ndim)))
    denominator = target.pow(2).sum(dim=tuple(range(1, target.ndim))).clamp_min(eps)
    return (numerator / denominator).sqrt()


def pearson_r(prediction, target, eps: float = 1.0e-8):
    pred = prediction.flatten(start_dim=1)
    tgt = target.flatten(start_dim=1)
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)
    numerator = (pred_centered * tgt_centered).sum(dim=1)
    denominator = pred_centered.norm(dim=1).clamp_min(eps) * tgt_centered.norm(dim=1).clamp_min(eps)
    return numerator / denominator


def mean_absolute_error(prediction, target):
    return (prediction - target).abs().mean(dim=tuple(range(1, prediction.ndim)))


def summarize_forward_metrics(prediction, target):
    if isinstance(prediction, dict):
        per_target = {
            name: summarize_forward_metrics(prediction[name], target[name])
            for name in sorted(prediction)
        }
        keys = ("relative_l2", "pearson_r", "mae")
        aggregate = {
            key: float(sum(metrics[key] for metrics in per_target.values()) / max(len(per_target), 1))
            for key in keys
        }
        return {
            **aggregate,
            "per_target": per_target,
        }

    rl2 = relative_l2(prediction, target).mean().item()
    r = pearson_r(prediction, target).mean().item()
    mae = mean_absolute_error(prediction, target).mean().item()
    return {
        "relative_l2": float(rl2),
        "pearson_r": float(r),
        "mae": float(mae),
    }


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        if self.count == 0:
            return math.nan
        return self.total / self.count


def build_forward_metric_meters(target_names: list[str] | None = None) -> dict:
    base = {
        "relative_l2": AverageMeter(),
        "pearson_r": AverageMeter(),
        "mae": AverageMeter(),
        "inference_time_ms": AverageMeter(),
    }
    if not target_names:
        return base
    base["per_target"] = {
        name: {
            "relative_l2": AverageMeter(),
            "pearson_r": AverageMeter(),
            "mae": AverageMeter(),
        }
        for name in target_names
    }
    return base


def update_forward_metric_meters(metric_meters: dict, batch_metrics: dict, batch_size: int):
    for key in ("relative_l2", "pearson_r", "mae"):
        metric_meters[key].update(batch_metrics[key], batch_size)
    if "per_target" not in metric_meters or "per_target" not in batch_metrics:
        return
    for target_name, per_target_metrics in batch_metrics["per_target"].items():
        for key in ("relative_l2", "pearson_r", "mae"):
            metric_meters["per_target"][target_name][key].update(per_target_metrics[key], batch_size)


def finalize_forward_metric_meters(metric_meters: dict) -> dict[str, float]:
    result = {
        "relative_l2": metric_meters["relative_l2"].average,
        "pearson_r": metric_meters["pearson_r"].average,
        "mae": metric_meters["mae"].average,
        "inference_time_ms": metric_meters["inference_time_ms"].average,
    }
    if "per_target" in metric_meters:
        result["per_target"] = {
            target_name: {
                key: meters[key].average
                for key in ("relative_l2", "pearson_r", "mae")
            }
            for target_name, meters in metric_meters["per_target"].items()
        }
    return result
