"""Validation-only AGS/DP-SGD screen at conventional MNIST privacy targets."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from common.accounting.accounting import calibrate_ags, calibrate_dpsgd
from common.accounting.artifacts import write_immutable_json
from common.accounting.config import ExperimentConfig
from common.accounting.model import build_model, parameter_count
from common.accounting.training import train_ags, train_dpsgd


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True, choices=("ags", "dpsgd"))
    parser.add_argument("--data", default="ags/mnist/accounting/data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        grid = yaml.safe_load(handle)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    raw = MNIST(args.data, train=True, download=True)
    inputs = raw.data[:, None].float().div_(255)
    inputs.sub_(grid["normalization_mean"]).div_(grid["normalization_std"])
    permutation = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(1729))
    validation_size = int(grid["validation_size"])
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    validation_set = TensorDataset(inputs[validation_indices], raw.targets[validation_indices])
    train_set = TensorDataset(inputs[train_indices], raw.targets[train_indices])
    validation_loader = DataLoader(validation_set, batch_size=512, shuffle=False)

    torch.manual_seed(int(grid["seed"]))
    template = build_model(grid["model"])
    initial_state = copy.deepcopy(template.state_dict())
    steps = math.ceil(len(train_set) / grid["batch_size"]) * grid["epochs"]
    q = grid["batch_size"] / len(train_set)
    caps = grid["ags_coordinate_clips"] if args.arm == "ags" else grid["dpsgd_l2_clips"]
    output = Path(args.output)
    checkpoint_dir = output.with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    calibrations = {}
    for epsilon in grid["epsilons"]:
        calibration_path = checkpoint_dir / f"epsilon_{epsilon}_calibration.json"
        if calibration_path.exists():
            with open(calibration_path, encoding="utf-8") as handle:
                calibration_payload = json.load(handle)
            from common.accounting.accounting import Calibration

            calibration = Calibration(**calibration_payload["calibration"])
            print(f"[{args.arm}-screen] reusing {calibration_path}", flush=True)
        else:
            print(f"[{args.arm}-screen] calibrating epsilon={epsilon}", flush=True)
            if args.arm == "ags":
                calibration = calibrate_ags(
                    epsilon,
                    grid["target_delta"],
                    coordinates=parameter_count(template),
                    steps=steps,
                    q=q,
                    mechanism="laplace",
                    interval=grid["ags_pld_interval"],
                    delta_fraction=grid["ags_delta_fraction"],
                )
            else:
                calibration = calibrate_dpsgd(
                    epsilon,
                    grid["target_delta"],
                    q=q,
                    steps=steps,
                    interval=grid["dpsgd_pld_interval"],
                    search_iterations=grid["dpsgd_search_iterations"],
                )
            write_immutable_json(
                calibration_path,
                {
                    "schema": 1,
                    "target_epsilon": epsilon,
                    "calibration": calibration.__dict__,
                },
            )
        calibrations[str(epsilon)] = calibration.__dict__
        for cap in caps:
            checkpoint = checkpoint_dir / f"epsilon_{epsilon}_cap_{cap}.json"
            if checkpoint.exists():
                with open(checkpoint, encoding="utf-8") as handle:
                    result = json.load(handle)["result"]
                results.append(result)
                print(f"[{args.arm}-screen] reusing {checkpoint}", flush=True)
                continue
            config = ExperimentConfig(
                experiment=grid["experiment"],
                model=grid["model"],
                seed=grid["seed"],
                batch_size=grid["batch_size"],
                epochs=grid["epochs"],
                learning_rate=grid["learning_rate"],
                target_epsilon=epsilon,
                target_delta=grid["target_delta"],
                beta_epsilon_ratio=grid["beta_epsilon_ratio"],
                coordinate_clip=cap if args.arm == "ags" else 0.1,
                l2_clip=cap if args.arm == "dpsgd" else 1.0,
                ags_mechanism="laplace",
                ags_release_mode="coordinate",
                sensitivity_method="direct_clipped_mean",
                ags_delta_fraction=grid["ags_delta_fraction"],
                pld_interval=(grid["ags_pld_interval"] if args.arm == "ags" else grid["dpsgd_pld_interval"]),
                gradient_chunk_size=grid["gradient_chunk_size"],
                accountant_search_iterations=grid["dpsgd_search_iterations"],
                arms=("ags_ss" if args.arm == "ags" else "dpsgd",),
            )
            model = build_model(grid["model"])
            model.load_state_dict(initial_state)
            if args.arm == "ags":
                result = train_ags(
                    model,
                    train_set,
                    validation_loader,
                    config,
                    calibration,
                    device,
                    use_gs=False,
                )
            else:
                result = train_dpsgd(model, train_set, validation_loader, config, calibration, device)
            result.update(
                {
                    "screen_arm": args.arm,
                    "target_epsilon": epsilon,
                    "clip": cap,
                    "model": grid["model"],
                    "evaluation_split": "validation",
                    "config": config.public_dict(),
                }
            )
            result["status"] = "completed" if math.isfinite(result["test_loss"]) else "diverged_nonfinite"
            result = json_safe(result)
            results.append(result)
            write_immutable_json(checkpoint, {"schema": 1, "result": result})
            loss_text = "nonfinite" if result["test_loss"] is None else f"{result['test_loss']:.5g}"
            clipping = result.get("coordinate_clipping_rate", result.get("example_l2_clipping_rate"))
            print(
                f"[{args.arm}-result] epsilon={epsilon} cap={cap} "
                f"accuracy={result['test_accuracy']:.4f} loss={loss_text} "
                f"clipping={clipping:.5f} status={result['status']}",
                flush=True,
            )
    digest = write_immutable_json(
        output,
        {
            "schema": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "grid": grid,
            "arm": args.arm,
            "calibrations": calibrations,
            "results": results,
            "note": "Exploratory validation-only clip marginalization; test set untouched.",
        },
    )
    print(f"wrote {output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
