#!/usr/bin/env python3
"""CI-style regression test for magnetic forward modeling.

What it checks
--------------
1) mode="prism_matched" produces stable reference numbers for a small deterministic case.
2) mode="standard_B" stays numerically sane (no blow-ups/NaNs) and remains close (within a
   loose tolerance) to prism_matched on the same case.
3) Optional smoke test: main.py correctly routes the `mode` parameter from config/CLI.

Usage
-----
python regression_ci.py
python regression_ci.py --baseline regression_baseline_mag.json
python regression_ci.py --smoke-main

Exit code
---------
0: pass
1: fail
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch

# Ensure local imports work when invoked from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mag_forward import forward_mag_tmi


def _metrics(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float32)
    return {
        "max": float(a.max()),
        "min": float(a.min()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "l2": float(np.sqrt(np.mean(a * a))),
    }


def _norm_mae(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.max(np.abs(b)) + 1e-12)
    return float(np.mean(np.abs(a - b)) / denom)


def _assert_close_abs(name: str, got: Dict[str, float], exp: Dict[str, float], tol: Dict[str, float]) -> Tuple[bool, str]:
    for k, exp_v in exp.items():
        got_v = got.get(k, None)
        if got_v is None:
            return False, f"[{name}] Missing metric key: {k}"
        t = float(tol.get(k, 0.0))
        if not math.isfinite(got_v):
            return False, f"[{name}] {k} is not finite: {got_v}"
        if abs(got_v - float(exp_v)) > t:
            return False, f"[{name}] {k} mismatch: got {got_v:.6g}, expected {float(exp_v):.6g} ± {t:.6g}"
    return True, ""


def _build_case(baseline: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    case = baseline["case"]
    nx, ny, nz = int(case["nx"]), int(case["ny"]), int(case["nz"])
    dx, dy, dz = float(case["dx"]), float(case["dy"]), float(case["dz"])

    model = np.zeros((nx, ny, nz), dtype=np.float32)
    blk = case["model"]["block"]
    x0, x1 = blk["x"]
    y0, y1 = blk["y"]
    z0, z1 = blk["z"]
    model[int(x0):int(x1), int(y0):int(y1), int(z0):int(z1)] = float(blk["value"])

    obs_conf = case["obs"]

    info = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "obs_conf": obs_conf,
        "heights_m": case.get("heights_m", [0.0]),
        "pad_factor": int(case.get("pad_factor", 2)),
        "input_type": case.get("input_type", "magnetization"),
        "output_unit": case.get("output_unit", "nt"),
    }

    return torch.from_numpy(model), info


def run_regression(baseline_path: str, device: str, smoke_main: bool) -> int:
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    t_model, info = _build_case(baseline)
    dev = torch.device(device)
    t_model = t_model.to(dev)

    # --- prism_matched ---
    prism_t, _ = forward_mag_tmi(
        t_model,
        info["dx"], info["dy"], info["dz"],
        heights_m=info["heights_m"],
        obs_conf=info["obs_conf"],
        input_type=info["input_type"],
        output_unit=info["output_unit"],
        mode="prism_matched",
    )
    prism = prism_t[0, 0].detach().cpu().numpy()

    # --- standard_B ---
    std_t, _ = forward_mag_tmi(
        t_model,
        info["dx"], info["dy"], info["dz"],
        heights_m=info["heights_m"],
        obs_conf=info["obs_conf"],
        input_type=info["input_type"],
        output_unit=info["output_unit"],
        pad_factor=info["pad_factor"],
        mode="standard_B",
    )
    std = std_t[0, 0].detach().cpu().numpy()

    # Sanity checks
    sanity = baseline["tolerances"]["sanity"]
    max_abs = float(sanity["max_abs_nT"])
    if sanity.get("no_nan", True):
        if not np.isfinite(prism).all():
            print("FAIL: prism_matched contains NaN/Inf")
            return 1
        if not np.isfinite(std).all():
            print("FAIL: standard_B contains NaN/Inf")
            return 1
    if float(np.max(np.abs(prism))) > max_abs:
        print(f"FAIL: prism_matched blew up (max_abs={np.max(np.abs(prism)):.3g} > {max_abs:.3g})")
        return 1
    if float(np.max(np.abs(std))) > max_abs:
        print(f"FAIL: standard_B blew up (max_abs={np.max(np.abs(std)):.3g} > {max_abs:.3g})")
        return 1

    # Metric comparisons
    got_prism = _metrics(prism)
    got_std = _metrics(std)

    exp_prism = baseline["expected"]["prism_matched"]
    exp_std = baseline["expected"]["standard_B"]

    tol_prism = baseline["tolerances"]["prism_matched_abs"]
    tol_std = baseline["tolerances"]["standard_B_abs"]

    ok, msg = _assert_close_abs("prism_matched", got_prism, exp_prism, tol_prism)
    if not ok:
        print("FAIL:", msg)
        print("Got prism metrics:", got_prism)
        return 1

    ok, msg = _assert_close_abs("standard_B", got_std, {k: exp_std[k] for k in ["max", "min", "mean", "std", "l2"]}, tol_std)
    if not ok:
        print("FAIL:", msg)
        print("Got std metrics:", got_std)
        return 1

    # Cross-metric: normalized MAE std vs prism
    nmae = _norm_mae(std, prism)
    nmae_max = float(baseline["tolerances"]["standard_B_norm_mae_max"])
    if nmae > nmae_max:
        print(f"FAIL: standard_B too far from prism_matched: norm_MAE={nmae:.6g} > {nmae_max:.6g}")
        return 1

    print("PASS: regression metrics OK")
    print("  prism_matched:", got_prism)
    print("  standard_B   :", got_std)
    print(f"  norm_MAE(std vs prism)={nmae:.6g}")

    # Optional: smoke test main.py routes mode from config/CLI
    if smoke_main:
        print("\nRunning main.py smoke tests...")
        tmp_out = os.path.join("output", "_ci_tmp.npz")
        cfg_path = os.path.join(os.path.dirname(baseline_path), "config.yaml")
        # Run standard_B via config
        subprocess.check_call([sys.executable, "main.py", "--config", cfg_path, "--mode", "standard_B"], cwd=os.path.dirname(__file__))
        # Run prism_matched via CLI override
        subprocess.check_call([sys.executable, "main.py", "--config", cfg_path, "--mode", "prism_matched"], cwd=os.path.dirname(__file__))
        print("PASS: main.py mode routing smoke test")

    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=str, default="regression_baseline_mag.json")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--smoke-main", action="store_true", help="Also run main.py to verify config/CLI mode routing")
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite expected metrics in the baseline file with current results (use only when changes are intentional).",
    )
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA requested but not available; falling back to CPU")
        args.device = "cpu"

    baseline_path = args.baseline
    if not os.path.exists(baseline_path):
        print(f"FAIL: baseline file not found: {baseline_path}")
        sys.exit(1)

    # Optional baseline update shortcut
    if args.update_baseline:
        # Run once and overwrite expected metrics.
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        t_model, info = _build_case(baseline)
        dev = torch.device(args.device)
        t_model = t_model.to(dev)

        prism_t, _ = forward_mag_tmi(
            t_model,
            info["dx"], info["dy"], info["dz"],
            heights_m=info["heights_m"],
            obs_conf=info["obs_conf"],
            input_type=info["input_type"],
            output_unit=info["output_unit"],
            mode="prism_matched",
        )
        prism = prism_t[0, 0].detach().cpu().numpy()

        std_t, _ = forward_mag_tmi(
            t_model,
            info["dx"], info["dy"], info["dz"],
            heights_m=info["heights_m"],
            obs_conf=info["obs_conf"],
            input_type=info["input_type"],
            output_unit=info["output_unit"],
            pad_factor=info["pad_factor"],
            mode="standard_B",
        )
        std = std_t[0, 0].detach().cpu().numpy()

        baseline["expected"]["prism_matched"] = _metrics(prism)
        baseline["expected"]["standard_B"] = {**_metrics(std), "norm_mae_vs_prism": _norm_mae(std, prism)}
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline updated: {baseline_path}")
        sys.exit(0)

    rc = run_regression(baseline_path, args.device, args.smoke_main)
    sys.exit(rc)


if __name__ == "__main__":
    main()
