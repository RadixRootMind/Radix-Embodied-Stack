#!/usr/bin/env python3
"""Prepare a trained LeRobot PI0.5 checkpoint for export or Houmo runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_json(path: Path, updates: dict[str, object]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="Checkpoint pretrained_model directory")
    parser.add_argument("--mode", choices=("export", "runtime"), required=True)
    parser.add_argument("--tokenizer", default="./models/paligemma-3b-pt-224")
    args = parser.parse_args()

    config = args.model / "config.json"
    preprocessor = args.model / "policy_preprocessor.json"
    if not config.is_file() or not preprocessor.is_file():
        raise SystemExit("Checkpoint must contain config.json and policy_preprocessor.json")

    update_json(
        config,
        {
            "type": "pi05" if args.mode == "export" else "hm_pi05",
            "device": "cpu",
            "compile_model": False,
        },
    )

    data = json.loads(preprocessor.read_text(encoding="utf-8"))

    def patch(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "device":
                    value[key] = "cpu"
                elif key == "tokenizer_name":
                    value[key] = args.tokenizer
                else:
                    patch(child)
        elif isinstance(value, list):
            for child in value:
                patch(child)

    patch(data)
    preprocessor.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {args.model} for {args.mode}")


if __name__ == "__main__":
    main()
