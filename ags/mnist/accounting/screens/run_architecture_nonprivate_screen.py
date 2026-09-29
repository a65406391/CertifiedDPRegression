"""Nonprivate validation screen for the public MNIST CNN candidates."""

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

from common.accounting.artifacts import write_immutable_json
from common.accounting.config import ExperimentConfig
from common.accounting.model import build_model, parameter_count
from common.accounting.training import train_nonprivate


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
    split_generator = torch.Generator().manual_seed(1729)
    permutation = torch.randperm(len(inputs), generator=split_generator)
    validation_size = int(grid["validation_size"])
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    validation_set = TensorDataset(inputs[validation_indices], raw.targets[validation_indices])
    train_set = TensorDataset(inputs[train_indices], raw.targets[train_indices])
    validation_loader = DataLoader(validation_set, batch_size=512, shuffle=False)

    output = Path(args.output)
    checkpoint_dir = output.with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in grid["models"]:
        torch.manual_seed(int(grid["seed"]))
        template = build_model(model_name)
        initial_state = copy.deepcopy(template.state_dict())
        for learning_rate in grid["learning_rates"]:
            checkpoint = checkpoint_dir / f"{model_name}_lr_{learning_rate}.json"
            if checkpoint.exists():
                with open(checkpoint, encoding="utf-8") as handle:
                    result = json.load(handle)["result"]
                results.append(result)
                print(f"[architecture] reusing {checkpoint}", flush=True)
                continue
            config = ExperimentConfig(
                experiment=grid["experiment"],
                model=model_name,
                seed=grid["seed"],
                batch_size=grid["batch_size"],
                epochs=grid["selection_epochs"],
                learning_rate=learning_rate,
                gradient_chunk_size=grid["gradient_chunk_size"],
                arms=("nonprivate",),
            )
            model = build_model(model_name)
            model.load_state_dict(initial_state)
            result = train_nonprivate(model, train_set, validation_loader, config, device)
            if not math.isfinite(result["test_loss"]):
                result["test_loss"] = None
                result["status"] = "diverged_nonfinite_loss"
            else:
                result["status"] = "completed"
            result.update(
                {
                    "model": model_name,
                    "parameter_count": parameter_count(model),
                    "learning_rate": learning_rate,
                    "evaluation_split": "validation",
                }
            )
            results.append(result)
            write_immutable_json(
                checkpoint,
                {
                    "schema": 1,
                    "grid": grid,
                    "config": config.public_dict(),
                    "result": result,
                },
            )
            print(
                f"[architecture-result] model={model_name} lr={learning_rate} "
                f"validation_accuracy={result['test_accuracy']:.4f}",
                flush=True,
            )
    digest = write_immutable_json(
        output,
        {
            "schema": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "grid": grid,
            "results": results,
            "note": "Architecture and LR selection uses only a deterministic validation split.",
        },
    )
    print(f"wrote {output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
