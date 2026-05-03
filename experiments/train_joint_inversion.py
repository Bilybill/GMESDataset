from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.datasets.benchmark_index import DEFAULT_PARTITION_ALIASES, build_forward_index
from experiments.datasets.gmes_inverse_dataset import AVAILABLE_MODALITIES, normalize_modalities
from experiments.datasets.gmes_joint_inversion_dataset import (
    GMESJointInversionDataset,
    TARGET_FIELDS,
    filter_records_for_joint_inversion,
)
from experiments.models.joint_inversion import LateFusionJointInversionModel
from experiments.utils.metrics_forward import (
    AverageMeter,
    build_forward_metric_meters,
    finalize_forward_metric_meters,
    summarize_forward_metrics,
    update_forward_metric_meters,
)
from experiments.utils.splits import split_records_by_background


def _resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is out of range.")
    return device


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_modalities_to_device(modalities: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        for name, tensor in modalities.items()
    }


def _target_tree(volume: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        field: volume[:, idx : idx + 1]
        for idx, field in enumerate(TARGET_FIELDS)
    }


def _format_metric_block(metrics: dict) -> dict:
    ordered = {}
    for key in ("mae", "pearson_r", "nonzero_relative_l2", "relative_l2", "inference_time_ms", "loss"):
        if key in metrics:
            ordered[key] = metrics[key]
    for key, value in metrics.items():
        if key == "per_target":
            continue
        if key not in ordered:
            ordered[key] = value
    if "per_target" in metrics:
        ordered["per_target"] = metrics["per_target"]
    return ordered


def run_epoch(model, loader, optimizer, device: torch.device, training: bool):
    model.train(training)
    loss_meter = AverageMeter()
    metrics_meter = None

    for batch in loader:
        modalities = _move_modalities_to_device(batch["modalities"], device)
        targets = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            predictions = model(modalities)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = int(targets.shape[0])
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(batch_size, 1)
        batch_metrics = summarize_forward_metrics(_target_tree(predictions.detach()), _target_tree(targets))
        if metrics_meter is None:
            metrics_meter = build_forward_metric_meters(target_names=list(TARGET_FIELDS))
        loss_meter.update(loss.item(), batch_size)
        update_forward_metric_meters(metrics_meter, batch_metrics, batch_size)
        metrics_meter["inference_time_ms"].update(elapsed_ms, batch_size)

    if metrics_meter is None:
        metrics_meter = build_forward_metric_meters(target_names=list(TARGET_FIELDS))
    result = {"loss": loss_meter.average}
    result.update(finalize_forward_metric_meters(metrics_meter))
    return result


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "PretrainDataset_forward"))
    default_output_root = os.path.abspath(os.path.join(script_dir, "..", "DATAFOLDER", "ExperimentRuns", "joint_inversion"))
    parser = argparse.ArgumentParser(description="Train GMES-3D joint multiphysics inversion baselines.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument("--include-top-levels", type=str, nargs="*", default=list(DEFAULT_PARTITION_ALIASES))
    parser.add_argument("--modalities", type=str, nargs="+", default=list(AVAILABLE_MODALITIES))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-base-channels", type=int, default=128)
    parser.add_argument("--target-shape", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--development-source-prefixes", type=str, nargs="*", default=None)
    parser.add_argument("--heldout-source-prefixes", type=str, nargs="*", default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-heldout-samples", type=int, default=0)
    parser.add_argument("--output-root", type=str, default=default_output_root)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    modalities = normalize_modalities(args.modalities)
    target_shape = tuple(int(x) for x in args.target_shape)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    records = build_forward_index(
        args.root,
        include_top_levels=args.include_top_levels,
        partition_aliases=DEFAULT_PARTITION_ALIASES,
        require_all_modalities=False,
        development_source_prefixes=args.development_source_prefixes,
        heldout_source_prefixes=args.heldout_source_prefixes,
    )
    records = filter_records_for_joint_inversion(records, modalities)
    train_records, val_records, heldout_records = split_records_by_background(
        records,
        validation_fraction=args.val_fraction,
        seed=args.seed,
    )
    if args.max_train_samples > 0:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_records = val_records[: args.max_val_samples]
    if args.max_heldout_samples > 0:
        heldout_records = heldout_records[: args.max_heldout_samples]
    if not train_records or not val_records or not heldout_records:
        raise RuntimeError("Train/val/held-out splits must all be non-empty.")

    train_dataset = GMESJointInversionDataset(train_records, modalities=modalities, target_shape=target_shape)
    val_dataset = GMESJointInversionDataset(val_records, modalities=modalities, target_shape=target_shape)
    heldout_dataset = GMESJointInversionDataset(heldout_records, modalities=modalities, target_shape=target_shape)

    model = LateFusionJointInversionModel(
        modalities=modalities,
        embedding_dim=args.embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        target_shape=target_shape,
        decoder_base_channels=args.decoder_base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    heldout_loader = DataLoader(heldout_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_name = "_".join(("joint_inversion", *modalities, "to", *TARGET_FIELDS))
    output_dir = os.path.abspath(os.path.join(args.output_root, run_name))
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")
    last_ckpt_path = os.path.join(output_dir, "last_model.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_epoch_record = None
    start_epoch = 1

    def refresh_best_from_history() -> None:
        nonlocal best_val, best_epoch, best_epoch_record
        best_val = float("inf")
        best_epoch = 0
        best_epoch_record = None
        for record in history:
            val_rl2 = float(record.get("val", {}).get("relative_l2", float("inf")))
            if val_rl2 < best_val:
                best_val = val_rl2
                best_epoch = int(record.get("epoch", 0))
                best_epoch_record = record

    def write_metrics_snapshot() -> None:
        if not history:
            return
        final_epoch_record = history[-1]
        results = {
            "task": "joint_inversion",
            "modalities": modalities,
            "target_fields": TARGET_FIELDS,
            "device": str(device),
            "splits": {
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "heldout_size": len(heldout_dataset),
            },
            "target_shape": target_shape,
            "paths": {
                "output_dir": output_dir,
                "best_checkpoint": best_ckpt_path,
                "last_checkpoint": last_ckpt_path,
            },
            "selection": {"metric": "val.relative_l2", "best_epoch": best_epoch},
            "summary": {
                "best_validation": _format_metric_block(best_epoch_record["val"]) if best_epoch_record else {},
                "heldout_at_best_validation": _format_metric_block(best_epoch_record["heldout"]) if best_epoch_record else {},
                "final_validation": _format_metric_block(final_epoch_record["val"]),
                "final_heldout": _format_metric_block(final_epoch_record["heldout"]),
            },
            "history": history,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=True, indent=2)

    if args.resume:
        if not os.path.exists(last_ckpt_path):
            raise FileNotFoundError(f"Cannot resume because last checkpoint does not exist: {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", len(history))) + 1
        refresh_best_from_history()
        print(f"[resume] Loaded checkpoint: {last_ckpt_path}")
        print(f"[resume] Continuing from epoch {start_epoch} / {args.epochs}. Best epoch so far: {best_epoch}")

    print("====== GMES-3D Joint Inversion Training ======")
    print(f"Modalities  : {modalities}")
    print(f"Targets     : {TARGET_FIELDS}")
    print(f"Target shape: {target_shape}")
    print(f"Device      : {device}")
    print(f"Train/Val   : {len(train_dataset)} / {len(val_dataset)}")
    print(f"Held-out    : {len(heldout_dataset)}")
    print(f"Output dir  : {output_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, training=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, training=False)
            heldout_metrics = run_epoch(model, heldout_loader, optimizer=None, device=device, training=False)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "heldout": heldout_metrics}
        history.append(record)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_rl2={val_metrics['relative_l2']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_r={val_metrics['pearson_r']:.6f} "
            f"heldout_rl2={heldout_metrics['relative_l2']:.6f} "
            f"heldout_mae={heldout_metrics['mae']:.6f} "
            f"heldout_r={heldout_metrics['pearson_r']:.6f}"
        )
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "history": history,
            "target_fields": TARGET_FIELDS,
            "target_shape": target_shape,
        }
        torch.save(state, last_ckpt_path)
        if val_metrics["relative_l2"] < best_val:
            best_val = val_metrics["relative_l2"]
            best_epoch = epoch
            best_epoch_record = record
            torch.save(state, best_ckpt_path)
        write_metrics_snapshot()

    write_metrics_snapshot()
    print(f"[+] Saved metrics to {metrics_path}")
    print(f"[+] Saved best checkpoint to {best_ckpt_path}")


if __name__ == "__main__":
    main()
