"""Run a privacy-calibrated experiment from one public configuration."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.metadata
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets

from common.accounting.accounting import calibrate_ags
from common.accounting.accounting import calibrate_dpsgd
from common.accounting.artifacts import write_immutable_json
from common.accounting.config import ExperimentConfig
from common.accounting.model import build_model
from common.accounting.model import parameter_count
from common.accounting.training import train_ags
from common.accounting.training import train_dpsgd
from common.accounting.training import train_nonprivate
from common.accounting.training import ags_release_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="ags/mnist/accounting/data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no GPU is visible to PyTorch. Launch the "
            "workspace/container with an NVIDIA device, then rerun unchanged."
        )
    device = torch.device(args.device)
    torch.manual_seed(config.seed)
    template = build_model(config.model)
    if config.model == "lenet5" and parameter_count(template) != 44_426:
        raise RuntimeError("LeNet-5 parameter count is not 44,426")
    initial_state = copy.deepcopy(template.state_dict())
    raw_train = datasets.MNIST(args.data, train=True, download=True)
    raw_test = datasets.MNIST(args.data, train=False, download=True)
    # This is exactly ToTensor's MNIST mapping, materialized once for fast indexed
    # batches: uint8 [0,255] -> float32 [0,1], with a channel dimension.
    train_set = TensorDataset(raw_train.data[:, None].float().div_(255), raw_train.targets)
    test_set = TensorDataset(raw_test.data[:, None].float().div_(255), raw_test.targets)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
    steps = math.ceil(len(train_set) / config.batch_size) * config.epochs
    q = config.batch_size / len(train_set)
    p = parameter_count(template)
    ags_coordinates = ags_release_count(template, config.ags_release_mode)
    calibrations = {}
    results = []
    for arm in config.arms:
        model = build_model(config.model)
        model.load_state_dict(initial_state)
        if arm == "nonprivate":
            results.append(train_nonprivate(model, train_set, test_loader, config, device))
        elif arm == "ags_ss":
            if "ags" not in calibrations:
                print("[accounting] calibrating conservative AGS PLD", flush=True)
                calibrations["ags"] = calibrate_ags(
                    config.target_epsilon,
                    config.target_delta,
                    coordinates=ags_coordinates,
                    steps=steps,
                    q=q,
                    mechanism=config.ags_mechanism,
                    interval=config.pld_interval,
                    delta_fraction=config.ags_delta_fraction,
                )
                print(f"[accounting] AGS {calibrations['ags']}", flush=True)
            results.append(
                train_ags(model, train_set, test_loader, config, calibrations["ags"], device, use_gs=False)
            )
        elif arm == "ags_gs":
            calibration_key = "ags_gs"
            if calibration_key not in calibrations:
                accounting_mechanism = "laplace_pure" if config.ags_mechanism == "laplace" else "cauchy"
                calibrations[calibration_key] = calibrate_ags(
                    config.target_epsilon,
                    config.target_delta,
                    coordinates=ags_coordinates,
                    steps=steps,
                    q=q,
                    mechanism=accounting_mechanism,
                    interval=config.pld_interval,
                    delta_fraction=config.ags_delta_fraction,
                )
            results.append(
                train_ags(
                    model,
                    train_set,
                    test_loader,
                    config,
                    calibrations[calibration_key],
                    device,
                    use_gs=True,
                )
            )
        elif arm == "dpsgd":
            if "dpsgd" not in calibrations:
                print("[accounting] calibrating replace-one sampled-Gaussian PLD", flush=True)
                calibrations["dpsgd"] = calibrate_dpsgd(
                    config.target_epsilon,
                    config.target_delta,
                    q=q,
                    steps=steps,
                    interval=config.pld_interval,
                    search_iterations=config.accountant_search_iterations,
                )
                print(f"[accounting] DP-SGD {calibrations['dpsgd']}", flush=True)
            results.append(train_dpsgd(model, train_set, test_loader, config, calibrations["dpsgd"], device))
        else:
            raise ValueError(f"unknown arm: {arm}")
        print(f"[complete] {results[-1]}", flush=True)
    payload = {
        "schema": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config.public_dict(),
        "results": results,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "dp-accounting", "opacus"]
        },
        "accounting_note": "AGS uses conservative generic PLDs; coordinates are composed before one WOR amplification.",
    }
    digest = write_immutable_json(Path(args.output), payload)
    print(f"wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
