#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/logs"
OUT_DIR="outputs/xh2"
mkdir -p "$LOG_DIR" "$OUT_DIR"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/so101_build_${STAMP}_${name}.log"
  echo "==== [$name] $*"
  "$@" 2>&1 | tee "$log"
}

link_runtime_names() {
  ln -sfn siglip_so101.hmm "$OUT_DIR/siglip.hmm"
  ln -sfn gemma_2b_prefill_so101.hmm "$OUT_DIR/gemma_2b_prefill.hmm"
  ln -sfn gemma_expert_300m_decode_so101.hmm "$OUT_DIR/gemma_expert_300m_decode.hmm"
  ln -sfn action_in_proj_so101.hmm "$OUT_DIR/action_in_proj.hmm"
  ln -sfn action_out_proj_so101.hmm "$OUT_DIR/action_out_proj.hmm"
  ln -sfn time_mlp_so101.hmm "$OUT_DIR/time_mlp.hmm"
}

check_runtime_files() {
  local missing=0
  local files=(
    siglip.hmm
    gemma_2b_prefill.hmm
    gemma_expert_300m_decode.hmm
    action_in_proj.hmm
    action_out_proj.hmm
    time_mlp.hmm
    embedding.pt
  )
  echo "Runtime artifact check:"
  for f in "${files[@]}"; do
    if [[ -e "$OUT_DIR/$f" ]]; then
      du -h "$OUT_DIR/$f" | sed 's/^/  /'
    else
      echo "  MISSING: $OUT_DIR/$f"
      missing=1
    fi
  done
  return "$missing"
}

echo "SO101 PI0.5 Houmo HMM build"
echo "  logs: $LOG_DIR/so101_build_${STAMP}_*.log"

run_step vision python3 -u build_vision_so101.py
run_step gemma_2b_prefill python3 -u build_gemma_2b_prefill_so101.py
run_step expert_decode python3 -u build_gemma_expert_300m_decode_so101.py
run_step action_time python3 -u build_pi05_action_time_linear_so101.py

link_runtime_names
check_runtime_files

echo "SUCCESS: SO101 build finished."
echo "Before running deployment inference, patch the model config for hm_pi05:"
echo "  python3 so101_patch_model_config.py --model-dir models/pretrained_model --mode infer"
