"""Audit public generic-PLD resolution and delta-allocation choices."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import yaml

from common.accounting.accounting import calibrate_ags
from common.accounting.artifacts import write_immutable_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        grid = yaml.safe_load(handle)
    output = Path(args.output)
    checkpoint_dir = output.with_suffix("")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for interval in grid["pld_intervals"]:
        for fraction in grid["delta_fractions"]:
            checkpoint = checkpoint_dir / f"interval_{interval}_fraction_{fraction}.json"
            if checkpoint.exists():
                with open(checkpoint, encoding="utf-8") as handle:
                    result = json.load(handle)["result"]
                results.append(result)
                print(f"[accountant-sweep] reusing {checkpoint}", flush=True)
                continue
            calibration = calibrate_ags(
                grid["target_epsilon"],
                grid["target_delta"],
                coordinates=grid["coordinates"],
                steps=grid["steps"],
                q=grid["q"],
                mechanism=grid["mechanism"],
                interval=interval,
                delta_fraction=fraction,
            )
            result = {
                "pld_interval": interval,
                "delta_fraction": fraction,
                **calibration.__dict__,
            }
            results.append(result)
            write_immutable_json(
                checkpoint,
                {
                    "schema": 1,
                    "grid": grid,
                    "result": result,
                },
            )
            print(
                f"[accountant-result] interval={interval} fraction={fraction} "
                f"eps_coord={calibration.mechanism_parameter:.8g} "
                f"epsilon={calibration.epsilon:.8g}",
                flush=True,
            )
    digest = write_immutable_json(
        output,
        {
            "schema": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "grid": grid,
            "results": results,
            "note": "Public accountant-only sweep; larger eps_coord means less AGS noise.",
        },
    )
    print(f"wrote {output} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
