#!/usr/bin/env python3
"""Patch SO101 PI0.5 model/config files for Houmo export or inference.

This script intentionally edits only lightweight config files. Model weights are not changed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = "models/pretrained_model"
DEFAULT_TOKENIZER_DIR = "models/paligemma-3b-pt-224"
DEFAULT_BASE_MODEL_DIR = "models/pi05_base"
GEMMA_CFG = ROOT / "config/pi0/llm/pi05_gemma_2b_xh2a_2k_so101_mask.py"
EXPERT_CFG = ROOT / "config/pi0/llm/pi05_gemma_expert_300m_xh2a_2k_so101_mask.py"


def _input(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def project_path_arg(value: str | None, default: str) -> str:
    value = (value or "").strip() or default
    return value[2:] if value.startswith("./") else value


def resolve_path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def backup(path: Path, stamp: str, enabled: bool) -> None:
    if not enabled or not path.exists():
        return
    dst = path.with_name(path.name + f".bak_{stamp}")
    if not dst.exists():
        shutil.copy2(path, dst)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a JSON object")
    return data


def dump_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_steps(data: dict[str, Any]):
    steps = data.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                yield step


def set_preprocessor(pre_path: Path, tokenizer_name: str, stamp: str, do_backup: bool) -> list[str]:
    data = load_json(pre_path)
    changed: list[str] = []

    for step in iter_steps(data):
        name = step.get("registry_name")
        cfg = step.setdefault("config", {})
        if not isinstance(cfg, dict):
            continue

        if name == "rename_observations_processor":
            rename_map = cfg.setdefault("rename_map", {})
            if isinstance(rename_map, dict) and rename_map.get("observation.images.top") != "observation.images.front":
                rename_map["observation.images.top"] = "observation.images.front"
                changed.append("rename observation.images.top -> observation.images.front")

        if name == "tokenizer_processor":
            if cfg.get("tokenizer_name") != tokenizer_name:
                cfg["tokenizer_name"] = tokenizer_name
                changed.append(f"tokenizer_name={tokenizer_name}")

        if name == "device_processor":
            if cfg.get("device") != "cpu":
                cfg["device"] = "cpu"
                changed.append("preprocessor device=cpu")

    if changed:
        backup(pre_path, stamp, do_backup)
        dump_json(pre_path, data)
    return changed


def set_postprocessor(post_path: Path, stamp: str, do_backup: bool) -> list[str]:
    data = load_json(post_path)
    changed: list[str] = []
    for step in iter_steps(data):
        if step.get("registry_name") == "device_processor":
            cfg = step.setdefault("config", {})
            if isinstance(cfg, dict) and cfg.get("device") != "cpu":
                cfg["device"] = "cpu"
                changed.append("postprocessor device=cpu")
    if changed:
        backup(post_path, stamp, do_backup)
        dump_json(post_path, data)
    return changed


def replace_assignment(text: str, name: str, value_literal: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*).*$", re.MULTILINE)
    replacement = rf"\g<1>{value_literal}"
    new_text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise ValueError(f"Could not find assignment for {name}")
    return new_text


def replace_kw_int(text: str, name: str, value: int) -> str:
    pattern = re.compile(rf"({re.escape(name)}\s*=\s*)\d+")
    new_text, count = pattern.subn(rf"\g<1>{value}", text)
    if count == 0:
        raise ValueError(f"Could not find keyword {name}")
    return new_text


def patch_llm_config(path: Path, model_dir: str, tokenizer_dir: str, input_sequence_length: int, stamp: str, do_backup: bool) -> list[str]:
    text = path.read_text(encoding="utf-8")
    new_text = text
    new_text = replace_assignment(new_text, "hf_model_dir", json.dumps(model_dir))
    new_text = replace_assignment(new_text, "config_dir", json.dumps(tokenizer_dir))
    new_text = replace_kw_int(new_text, "max_sequence_length", 1024)
    new_text = replace_kw_int(new_text, "input_sequence_length", input_sequence_length)
    if new_text != text:
        backup(path, stamp, do_backup)
        path.write_text(new_text, encoding="utf-8")
        return [f"{path.relative_to(ROOT)} -> hf_model_dir={model_dir}, config_dir={tokenizer_dir}, input_sequence_length={input_sequence_length}"]
    return []


def patch_model_config(config_path: Path, mode: str, n_action_steps: int, base_model_dir: str | None, stamp: str, do_backup: bool) -> list[str]:
    cfg = load_json(config_path)
    changed: list[str] = []

    # LeRobot 0.6 training configs include these runtime fields, but the Houmo
    # vendor PI05Config in this project rejects them during export/inference.
    unsupported_for_houmo_pi05 = [
        "pretrained_revision",
        "use_relative_actions",
        "relative_exclude_joints",
        "action_feature_names",
    ]
    for key in unsupported_for_houmo_pi05:
        if key in cfg:
            cfg.pop(key)
            changed.append(f"removed unsupported {key}")

    target_type = "hm_pi05" if mode == "infer" else "pi05"
    if cfg.get("type") != target_type:
        cfg["type"] = target_type
        changed.append(f"type={target_type}")

    if cfg.get("device") != "cpu":
        cfg["device"] = "cpu"
        changed.append("device=cpu")

    if cfg.get("compile_model") is not False:
        cfg["compile_model"] = False
        changed.append("compile_model=false")

    if cfg.get("n_action_steps") != n_action_steps:
        cfg["n_action_steps"] = n_action_steps
        changed.append(f"n_action_steps={n_action_steps}")

    if base_model_dir:
        old_base = str(cfg.get("pretrained_path", ""))
        if old_base != base_model_dir:
            cfg["pretrained_path"] = base_model_dir
            changed.append(f"pretrained_path={base_model_dir}")

    action_shape = None
    for key in ("output_features", "features"):
        section = cfg.get(key)
        if isinstance(section, dict) and isinstance(section.get("action"), dict):
            action_shape = section["action"].get("shape")
    if action_shape and action_shape != [6]:
        print(f"WARNING: action shape is {action_shape}, expected [6] for SO101.", file=sys.stderr)

    if changed:
        backup(config_path, stamp, do_backup)
        dump_json(config_path, cfg)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch SO101 PI0.5 model configs for Houmo export/inference.")
    parser.add_argument("--model-dir", help=f"trained model directory, default: {DEFAULT_MODEL_DIR}")
    parser.add_argument("--mode", choices=["export", "infer"], help="export keeps type=pi05; infer sets type=hm_pi05")
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_DIR, help="local PaliGemma tokenizer/config path")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_DIR, help="local pi05_base path written to pretrained_path; use empty string to skip")
    parser.add_argument("--n-action-steps", type=int, default=30)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    model_dir_str = project_path_arg(args.model_dir or _input("Model dir", DEFAULT_MODEL_DIR), DEFAULT_MODEL_DIR)
    mode = args.mode or _input("Mode: export or infer", "infer")
    if mode not in {"export", "infer"}:
        raise SystemExit("mode must be export or infer")

    tokenizer_str = project_path_arg(args.tokenizer_path, DEFAULT_TOKENIZER_DIR)
    base_model_str = project_path_arg(args.base_model_path, DEFAULT_BASE_MODEL_DIR) if args.base_model_path else ""
    model_dir = resolve_path(model_dir_str)
    if not model_dir.exists():
        raise SystemExit(f"Model dir does not exist: {model_dir}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    do_backup = not args.no_backup
    all_changes: list[str] = []

    config_path = model_dir / "config.json"
    pre_path = model_dir / "policy_preprocessor.json"
    post_path = model_dir / "policy_postprocessor.json"
    for required in (config_path, pre_path, post_path, GEMMA_CFG, EXPERT_CFG):
        if not required.exists():
            raise SystemExit(f"Required file missing: {required}")

    # Parse all checkpoint JSON before any config writes or backups.
    for path in (config_path, pre_path, post_path):
        load_json(path)

    all_changes += [f"config.json: {x}" for x in patch_model_config(config_path, mode, args.n_action_steps, base_model_str, stamp, do_backup)]
    all_changes += [f"policy_preprocessor.json: {x}" for x in set_preprocessor(pre_path, tokenizer_str, stamp, do_backup)]
    all_changes += [f"policy_postprocessor.json: {x}" for x in set_postprocessor(post_path, stamp, do_backup)]
    all_changes += patch_llm_config(GEMMA_CFG, model_dir_str, tokenizer_str, 712, stamp, do_backup)
    all_changes += patch_llm_config(EXPERT_CFG, model_dir_str, tokenizer_str, 50, stamp, do_backup)

    print("SO101 Houmo config patch complete")
    print(f"  model_dir: {model_dir_str}")
    print(f"  mode: {mode}")
    print(f"  tokenizer: {tokenizer_str}")
    print(f"  base_model: {base_model_str or '(unchanged)'}")
    print(f"  backup: {'enabled' if do_backup else 'disabled'}")
    if all_changes:
        print("Changes:")
        for item in all_changes:
            print(f"  - {item}")
    else:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
