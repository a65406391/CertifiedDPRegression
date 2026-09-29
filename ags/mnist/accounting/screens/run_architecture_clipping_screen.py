"""Short validation screen for noiseless clipping on MNIST CNN candidates."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from common.accounting.artifacts import write_immutable_json
from common.accounting.config import ExperimentConfig
from common.accounting.model import build_model, parameter_count
from common.accounting.training import train_clipped_noiseless


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="ags/mnist/accounting/data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", choices=("sample_cnn", "compact_selu_cnn"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--coordinate-cap", type=float)
    parser.add_argument("--l2-cap", type=float)
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
    selected_models = [args.model] if args.model else grid["models"]
    epochs = args.epochs if args.epochs is not None else grid["clip_screen_epochs"]
    coordinate_caps = [args.coordinate_cap] if args.coordinate_cap is not None else grid["coordinate_clips"]
    l2_caps = [args.l2_cap] if args.l2_cap is not None else grid["l2_clips"]
    for model_name in selected_models:
        torch.manual_seed(int(grid["seed"]))
        template = build_model(model_name)
        initial_state = copy.deepcopy(template.state_dict())
        for geometry, caps in (
            ("coordinate", coordinate_caps),
            ("l2", l2_caps),
        ):
            for cap in caps:
                checkpoint = checkpoint_dir / f"{model_name}_{geometry}_{cap}.json"
                if checkpoint.exists():
                    with open(checkpoint, encoding="utf-8") as handle:
                        result = json.load(handle)["result"]
                    results.append(result)
                    print(f"[clip-screen] reusing {checkpoint}", flush=True)
                    continue
                config = ExperimentConfig(
                    experiment=grid["experiment"],
                    model=model_name,
                    seed=grid["seed"],
                    batch_size=grid["batch_size"],
                    epochs=epochs,
                    learning_rate=0.1,
                    coordinate_clip=cap if geometry == "coordinate" else 0.1,
                    l2_clip=cap if geometry == "l2" else 1.0,
                    gradient_chunk_size=grid["gradient_chunk_size"],
                    arms=(f"clipped_{geometry}",),
                )
                model = build_model(model_name)
                model.load_state_dict(initial_state)
                result = train_clipped_noiseless(
                    model,
                    train_set,
                    validation_loader,
                    config,
                    device,
                    geometry=geometry,
                )
                result.update(
                    {
                        "model": model_name,
                        "parameter_count": parameter_count(model),
                        "geometry": geometry,
                        "clip": cap,
                        "learning_rate": 0.1,
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
                    f"[clip-result] model={model_name} geometry={geometry} cap={cap} "
                    f"accuracy={result['test_accuracy']:.4f} "
                    f"clip_rate={result['clipping_rate']:.4f}",
                    flush=True,
                )
    digest = write_immutable_json(
        output,
        {
            "schema": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "grid": grid,
            "results": results,
            "note": f"{epochs}-epoch public clipping screen on validation data; no noise.",
        },
    )
    print(f"wrote {output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
