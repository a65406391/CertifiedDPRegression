"""Run strict, privacy-matched multi-seed logistic finalists."""

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
    p = parameter_count(build_model("logistic"))

    ags_steps = math.ceil(len(train_set) / grid["ags_batch_size"]) * grid["epochs"]
    ags_q = grid["ags_batch_size"] / len(train_set)
    print("[strict] calibrating AGS accountant", flush=True)
    ags_calibration = calibrate_ags(
        grid["target_epsilon"],
        grid["target_delta"],
        coordinates=p,
        steps=ags_steps,
        q=ags_q,
        mechanism="laplace",
        interval=grid["pld_interval"],
        delta_fraction=grid["ags_delta_fraction"],
    )
    dp_steps = math.ceil(len(train_set) / grid["dpsgd_batch_size"]) * grid["epochs"]
    dp_q = grid["dpsgd_batch_size"] / len(train_set)
    print("[strict] calibrating DP-SGD accountant", flush=True)
    dp_calibration = calibrate_dpsgd(
        grid["target_epsilon"],
        grid["target_delta"],
        q=dp_q,
        steps=dp_steps,
        interval=grid["pld_interval"],
        search_iterations=grid["accountant_search_iterations"],
    )

    output = Path(args.output)
    checkpoint_dir = output.with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in grid["seeds"]:
        torch.manual_seed(seed)
        initial_state = copy.deepcopy(build_model("logistic").state_dict())
        for arm in ("ags_ss", "dpsgd"):
            checkpoint = checkpoint_dir / f"seed_{seed}_{arm}.json"
            if checkpoint.exists():
                with open(checkpoint, encoding="utf-8") as handle:
                    result = json.load(handle)["result"]
                results.append(result)
                print(f"[strict] reusing {checkpoint}", flush=True)
                continue
            batch_size = grid["ags_batch_size"] if arm == "ags_ss" else grid["dpsgd_batch_size"]
            config = ExperimentConfig(
                experiment=grid["experiment"],
                model="logistic",
                seed=seed,
                batch_size=batch_size,
                epochs=grid["epochs"],
                learning_rate=grid["learning_rate"],
                target_epsilon=grid["target_epsilon"],
                target_delta=grid["target_delta"],
                beta_epsilon_ratio=grid["laplace_beta_epsilon_ratio"],
                coordinate_clip=grid["coordinate_clip"],
                coordinate_clip_weight=grid["coordinate_clip_weight"],
                coordinate_clip_bias=grid["coordinate_clip_bias"],
                l2_clip=grid["l2_clip"],
                ags_mechanism="laplace",
                ags_release_mode="coordinate",
                sensitivity_method="direct_clipped_mean",
                ags_delta_fraction=grid["ags_delta_fraction"],
                pld_interval=grid["pld_interval"],
                gradient_chunk_size=grid["gradient_chunk_size"],
                accountant_search_iterations=grid["accountant_search_iterations"],
                arms=(arm,),
            )
            model = build_model("logistic")
            model.load_state_dict(initial_state)
            if arm == "ags_ss":
                result = train_ags(
                    model,
                    train_set,
                    test_loader,
                    config,
                    ags_calibration,
                    device,
                    use_gs=False,
                )
            else:
                result = train_dpsgd(model, train_set, test_loader, config, dp_calibration, device)
            result.update({"seed": seed, "strict_arm": arm, "config": config.public_dict()})
            results.append(result)
            write_immutable_json(checkpoint, {"schema": 1, "result": result})
            print(
                f"[strict-result] seed={seed} arm={arm} "
                f"accuracy={result['test_accuracy']:.4f} loss={result['test_loss']:.5g}",
                flush=True,
            )
    digest = write_immutable_json(
        output,
        {
            "schema": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "grid": grid,
            "calibrations": {
                "ags": ags_calibration.__dict__,
                "dpsgd": dp_calibration.__dict__,
            },
            "results": results,
            "note": "Strict PLD resolution and search; public finalists fixed before this run.",
        },
    )
    print(f"wrote {output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
