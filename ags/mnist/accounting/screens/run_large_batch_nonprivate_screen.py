"""Find stable learning rates for large/full-batch compact-CNN training."""

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
from common.accounting.model import build_model
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
    output = Path(args.output)
    checkpoint_dir = output.with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for batch_size in grid["batch_sizes"]:
        for learning_rate in grid["learning_rates"]:
            checkpoint = checkpoint_dir / f"batch_{batch_size}_lr_{learning_rate}.json"
            if checkpoint.exists():
                with open(checkpoint, encoding="utf-8") as handle:
                    result = json.load(handle)["result"]
                results.append(result)
                continue
            config = ExperimentConfig(
                experiment=grid["experiment"],
                model=grid["model"],
                seed=grid["seed"],
                batch_size=batch_size,
                epochs=grid["epochs"],
                learning_rate=learning_rate,
                arms=("nonprivate",),
            )
            model = build_model(grid["model"])
            model.load_state_dict(initial_state)
            result = train_nonprivate(model, train_set, validation_loader, config, device)
            finite = math.isfinite(result["test_loss"])
            if not finite:
                result["test_loss"] = None
            result.update(
                {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "status": "completed" if finite else "diverged_nonfinite",
                    "evaluation_split": "validation",
                }
            )
            results.append(result)
            write_immutable_json(checkpoint, {"schema": 1, "result": result})
            print(
                f"[large-batch-result] batch={batch_size} lr={learning_rate} "
                f"accuracy={result['test_accuracy']:.4f} status={result['status']}",
                flush=True,
            )
    digest = write_immutable_json(
        output,
        {
            "schema": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "grid": grid,
            "results": results,
            "note": "Validation-only nonprivate large-batch learning-rate screen.",
        },
    )
    print(f"wrote {output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
