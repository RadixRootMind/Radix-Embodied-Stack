#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  read -r -p "Model dir [models/pretrained_model]: " MODEL_DIR
  MODEL_DIR="${MODEL_DIR:-models/pretrained_model}"
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-models/paligemma-3b-pt-224}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-models/pi05_base}"
QUANT_TYPE="${QUANT_TYPE:-w8a8_sefp}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

patch_quant_type() {
  python3 - "$QUANT_TYPE" "$STAMP" pi05_export_vision_xh2a_so101.py pi05_export_other_xh2a_libero.py <<'PY'
import re
import shutil
import sys
from pathlib import Path

quant_type = sys.argv[1]
stamp = sys.argv[2]
for raw in sys.argv[3:]:
    path = Path(raw)
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(r'quant_type\s*=\s*["\'][^"\']+["\']', f'quant_type = "{quant_type}"', text)
    if count == 0:
        raise SystemExit(f"Could not find quant_type assignment in {path}")
    if new_text != text:
        shutil.copy2(path, path.with_name(path.name + f".bak_{stamp}"))
        path.write_text(new_text, encoding="utf-8")
        print(f"patched {path}: quant_type={quant_type}")
PY
}

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/so101_export_${STAMP}_${name}.log"
  echo "==== [$name] $*"
  "$@" 2>&1 | tee "$log"
}

echo "SO101 PI0.5 Houmo quantized export"
echo "  model_dir: $MODEL_DIR"
echo "  tokenizer: $TOKENIZER_PATH"
echo "  base_model: $BASE_MODEL_PATH"
echo "  quant_type: $QUANT_TYPE"
echo "  logs: $LOG_DIR/so101_export_${STAMP}_*.log"

python3 so101_patch_model_config.py \
  --model-dir "$MODEL_DIR" \
  --mode export \
  --tokenizer-path "$TOKENIZER_PATH" \
  --base-model-path "$BASE_MODEL_PATH"

patch_quant_type

run_step vision python3 -u pi05_export_vision_xh2a_so101.py --model_path "$MODEL_DIR"
run_step gemma_2b python3 -u pi05_export_llm_xh2a_libero.py --config ./config/pi0/llm/pi05_gemma_2b_xh2a_2k_so101_mask.py
run_step expert_300m python3 -u pi05_export_experts_xh2a_libero.py --config ./config/pi0/llm/pi05_gemma_expert_300m_xh2a_2k_so101_mask.py
run_step action_time python3 -u pi05_export_other_xh2a_libero.py --model_path "$MODEL_DIR"

echo "SUCCESS: SO101 quantized export finished."
echo "Next: bash so101_build_hmm.sh"
