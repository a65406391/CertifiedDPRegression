"""Run the public one-epoch logistic batch-size/learning-rate sweep."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

from common.accounting.accounting import calibrate_ags
from common.accounting.accounting import calibrate_dpsgd
from common.accounting.artifacts import write_immutable_json
from common.accounting.config import ExperimentConfig
from common.accounting.model import build_model
from common.accounting.model import parameter_count
from common.accounting.training import train_ags
from common.accounting.training import train_dpsgd


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
    p = parameter_count(template)
    results = []
    calibrations = {}
    for batch_size in grid["batch_size"]:
        output_path = Path(args.output)
        batch_path = output_path.with_name(f"{output_path.stem}_b{batch_size}{output_path.suffix}")
        if batch_path.exists():
            with open(batch_path, encoding="utf-8") as handle:
                completed = json.load(handle)
            results.extend(completed["results"])
            calibrations[str(batch_size)] = completed["calibrations"]
            print(f"[sweep] reusing immutable checkpoint {batch_path}", flush=True)
            continue
        steps = math.ceil(len(train_set) / batch_size) * int(grid["epochs"])
        q = batch_size / len(train_set)
        print(f"[sweep] batch_size={batch_size} steps={steps}", flush=True)
        laplace_calibration = calibrate_ags(
            grid["target_epsilon"],
            grid["target_delta"],
            coordinates=p,
            steps=steps,
            q=q,
            mechanism="laplace",
            interval=grid["pld_interval"],
            delta_fraction=grid["ags_delta_fraction"],
        )
        dpsgd_calibration = calibrate_dpsgd(
            grid["target_epsilon"],
            grid["target_delta"],
            q=q,
            steps=steps,
            interval=grid["pld_interval"],
            search_iterations=grid["accountant_search_iterations"],
        )
        calibrations[str(batch_size)] = {
            "laplace": laplace_calibration.__dict__,
            "dpsgd": dpsgd_calibration.__dict__,
        }
        for learning_rate in grid["learning_rate"]:
            common = ExperimentConfig(
                experiment=grid["experiment"],
                model="logistic",
                seed=grid["seed"],
                batch_size=batch_size,
                epochs=grid["epochs"],
                learning_rate=learning_rate,
                target_epsilon=grid["target_epsilon"],
                target_delta=grid["target_delta"],
                coordinate_clip=grid["coordinate_clip"],
                l2_clip=grid["l2_clip"],
                ags_mechanism="laplace",
                sensitivity_method="direct_clipped_mean",
                beta_epsilon_ratio=grid["laplace_beta_epsilon_ratio"],
                ags_delta_fraction=grid["ags_delta_fraction"],
                pld_interval=grid["pld_interval"],
                gradient_chunk_size=grid["gradient_chunk_size"],
                accountant_search_iterations=grid["accountant_search_iterations"],
            )
            for arm in grid["arms"]:
                model = build_model("logistic")
                model.load_state_dict(initial_state)
                if arm == "direct_laplace_ss":
                    result = train_ags(
                        model,
                        train_set,
                        test_loader,
                        common,
                        laplace_calibration,
                        device,
                        use_gs=False,
                    )
                elif arm == "laplace_gs":
                    result = train_ags(
                        model,
                        train_set,
                        test_loader,
                        common,
                        laplace_calibration,
                        device,
                        use_gs=True,
                    )
                elif arm == "dpsgd":
                    result = train_dpsgd(model, train_set, test_loader, common, dpsgd_calibration, device)
                else:
                    raise ValueError(f"unknown arm {arm}")
                result.update(
                    {
                        "sweep_arm": arm,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                    }
                )
                results.append(result)
                print(
                    f"[sweep-result] b={batch_size} lr={learning_rate} arm={arm} "
                    f"accuracy={result['test_accuracy']:.4f} loss={result['test_loss']:.5g}",
                    flush=True,
                )
        batch_results = [r for r in results if r["batch_size"] == batch_size]
        write_immutable_json(
            batch_path,
            {
                "schema": 1,
                "grid": grid,
                "batch_size": batch_size,
                "calibrations": calibrations[str(batch_size)],
                "results": batch_results,
            },
        )
        print(f"[sweep] checkpointed {batch_path}", flush=True)
    payload = {
        "schema": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "grid": grid,
        "calibrations": calibrations,
        "results": results,
        "note": "Exploratory five-iteration DP-SGD calibration and PLD interval 1e-3.",
    }
    digest = write_immutable_json(Path(args.output), payload)
    print(f"wrote {args.output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
