"""Private AGS coordinate-cap screen for the selected compact MNIST CNN."""

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

from common.accounting.accounting import calibrate_ags
from common.accounting.artifacts import write_immutable_json
from common.accounting.config import ExperimentConfig
from common.accounting.model import build_model, parameter_count
from common.accounting.training import train_ags


def json_safe(value):
    """Replace non-finite diagnostics with null while preserving the failed run."""
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
    print(
        f"[ags-clip-screen] calibrating coordinates={parameter_count(template)} steps={steps} q={q:.6g}",
        flush=True,
    )
    calibration = calibrate_ags(
        grid["target_epsilon"],
        grid["target_delta"],
        coordinates=parameter_count(template),
        steps=steps,
        q=q,
        mechanism="laplace",
        interval=grid["pld_interval"],
        delta_fraction=grid["ags_delta_fraction"],
    )
    print(f"[ags-clip-screen] calibration={calibration}", flush=True)

    output = Path(args.output)
    checkpoint_dir = output.with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for cap in grid["coordinate_clips"]:
        checkpoint = checkpoint_dir / f"coordinate_{cap}.json"
        if checkpoint.exists():
            with open(checkpoint, encoding="utf-8") as handle:
                result = json.load(handle)["result"]
            results.append(result)
            print(f"[ags-clip-screen] reusing {checkpoint}", flush=True)
            continue
        config = ExperimentConfig(
            experiment=grid["experiment"],
            model=grid["model"],
            seed=grid["seed"],
            batch_size=grid["batch_size"],
            epochs=grid["epochs"],
            learning_rate=grid["learning_rate"],
            target_epsilon=grid["target_epsilon"],
            target_delta=grid["target_delta"],
            beta_epsilon_ratio=grid["beta_epsilon_ratio"],
            coordinate_clip=cap,
            ags_mechanism="laplace",
            ags_release_mode="coordinate",
            sensitivity_method="direct_clipped_mean",
            ags_delta_fraction=grid["ags_delta_fraction"],
            pld_interval=grid["pld_interval"],
            gradient_chunk_size=grid["gradient_chunk_size"],
            arms=("ags_ss",),
        )
        model = build_model(grid["model"])
        model.load_state_dict(initial_state)
        result = train_ags(
            model,
            train_set,
            validation_loader,
            config,
            calibration,
            device,
            use_gs=False,
        )
        result.update(
            {
                "model": grid["model"],
                "coordinate_clip": cap,
                "evaluation_split": "validation",
                "config": config.public_dict(),
            }
        )
        finite_loss = math.isfinite(result["test_loss"])
        result["status"] = "completed" if finite_loss else "diverged_nonfinite"
        result = json_safe(result)
        results.append(result)
        write_immutable_json(checkpoint, {"schema": 1, "result": result})
        loss_text = "nonfinite" if result["test_loss"] is None else f"{result['test_loss']:.5g}"
        print(
            f"[ags-cap-result] cap={cap} accuracy={result['test_accuracy']:.4f} "
            f"loss={loss_text} status={result['status']} "
            f"clipping={result['coordinate_clipping_rate']:.5f}",
            flush=True,
        )
    digest = write_immutable_json(
        output,
        {
            "schema": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "grid": grid,
            "calibration": calibration.__dict__,
            "results": results,
            "note": "Exploratory validation-only AGS clipping screen; test set untouched.",
        },
    )
    print(f"wrote {output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
