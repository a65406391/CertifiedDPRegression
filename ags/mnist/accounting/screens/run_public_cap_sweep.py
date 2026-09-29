"""Explore a predeclared public weight/bias coordinate-cap grid."""

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

    raw_train = MNIST(args.data, train=True, download=True)
    raw_test = MNIST(args.data, train=False, download=True)
    train_set = TensorDataset(raw_train.data[:, None].float().div_(255), raw_train.targets)
    test_set = TensorDataset(raw_test.data[:, None].float().div_(255), raw_test.targets)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
    torch.manual_seed(int(grid["seed"]))
    template = build_model("logistic")
    initial_state = copy.deepcopy(template.state_dict())
    steps = math.ceil(len(train_set) / int(grid["batch_size"])) * int(grid["epochs"])
    q = int(grid["batch_size"]) / len(train_set)
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

    results = []
    output_path = Path(args.output)
    checkpoint_dir = output_path.with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for weight_cap in grid["weight_caps"]:
        for bias_cap in grid["bias_caps"]:
            checkpoint = checkpoint_dir / f"weight_{weight_cap}_bias_{bias_cap}.json"
            if checkpoint.exists():
                with open(checkpoint, encoding="utf-8") as handle:
                    result = json.load(handle)["result"]
                results.append(result)
                print(f"[cap-sweep] reusing {checkpoint}", flush=True)
                continue
            config = ExperimentConfig(
                experiment=grid["experiment"],
                model="logistic",
                seed=grid["seed"],
                batch_size=grid["batch_size"],
                epochs=grid["epochs"],
                learning_rate=grid["learning_rate"],
                target_epsilon=grid["target_epsilon"],
                target_delta=grid["target_delta"],
                coordinate_clip=grid["coordinate_clip"],
                coordinate_clip_weight=weight_cap,
                coordinate_clip_bias=bias_cap,
                l2_clip=grid["l2_clip"],
                ags_mechanism="laplace",
                sensitivity_method="direct_clipped_mean",
                beta_epsilon_ratio=grid["laplace_beta_epsilon_ratio"],
                ags_delta_fraction=grid["ags_delta_fraction"],
                pld_interval=grid["pld_interval"],
                gradient_chunk_size=grid["gradient_chunk_size"],
                accountant_search_iterations=grid["accountant_search_iterations"],
                arms=("ags_ss",),
            )
            model = build_model("logistic")
            model.load_state_dict(initial_state)
            result = train_ags(model, train_set, test_loader, config, calibration, device, use_gs=False)
            result.update({"weight_cap": weight_cap, "bias_cap": bias_cap})
            results.append(result)
            write_immutable_json(
                checkpoint,
                {
                    "schema": 1,
                    "grid": grid,
                    "config": config.public_dict(),
                    "calibration": calibration.__dict__,
                    "result": result,
                },
            )
            print(
                f"[cap-result] weight={weight_cap} bias={bias_cap} "
                f"accuracy={result['test_accuracy']:.4f} loss={result['test_loss']:.5g}",
                flush=True,
            )
    payload = {
        "schema": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "grid": grid,
        "calibration": calibration.__dict__,
        "results": results,
        "note": (
            "Exploratory public cap grid; selection is not a private-data tuning protocol. "
            "Generic AGS PLD accounting is conservative."
        ),
    }
    digest = write_immutable_json(output_path, payload)
    print(f"wrote {output_path} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
