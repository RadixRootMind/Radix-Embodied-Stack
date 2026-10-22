# PI0.5 on Houmo M50 for SO101

**From a LeRobot PI0.5 checkpoint to Houmo W8A8 HMONNX, M50 HMM, and SO101 closed-loop inference.**

[![LeRobot](https://img.shields.io/badge/Based%20on-LeRobot-blue)](https://github.com/huggingface/lerobot)
[![Target](https://img.shields.io/badge/Target-Houmo%20M50-green)](#supported-platforms)
[![Host](https://img.shields.io/badge/Host-Hygon%20C86--4G-red)](#supported-platforms)
[![Robot](https://img.shields.io/badge/Robot-SO101-orange)](#so101-inference)

[Overview](#overview) | [Platforms](#supported-platforms) | [Layout](#repository-layout) | [Prepare](#checkpoint-preparation) | [Export](#w8a8-export) | [Build](#hmm-build) | [Inference](#so101-inference) | [License](#acknowledgements-and-license)

> **Status:** This code was organized from a deployed SO101 real-robot environment. Hardware results are not claimed in this repository.

## Overview

This repository packages the PI0.5 deployment path used on a Hygon C86-4G host
with a Houmo M50 accelerator and an SO101 follower arm. It includes the local
LeRobot source, Houmo model-zoo integration, checkpoint patching utilities,
W8A8 export scripts, M50 HMM build scripts, and a real-robot inference loop.

The trained PI0.5 checkpoint is still required for model configuration,
tokenizer/preprocessor/postprocessor metadata, and normalization statistics.
Neural-network inference is performed by compiled Houmo `.hmm` artifacts.

### Demo Videos

<table>
  <tr>
    <td align="center" width="50%">
      <video src="https://github.com/user-attachments/assets/07e20a37-00ba-4bf3-9f65-1b1cb7fafeb7" controls width="420"></video>
    </td>
    <td align="center" width="50%">
      <video src="https://github.com/user-attachments/assets/00bbd130-dbaf-4592-9b34-f6b7890e6e32" controls width="420"></video>
    </td>
  </tr>
</table>

**Key Features**

- 🔧 Patch LeRobot PI0.5 checkpoint JSON for Houmo export or inference.
- ⚡ Export PI0.5 submodules to Houmo HMONNX with W8A8 PTQ.
- 🧱 Build M50 `.hmm` runtime artifacts from exported modules.
- 📷 Read SO101 top/side cameras plus joint state.
- 🦾 Run closed-loop SO101 inference and optionally send motor actions.

## Supported Platforms

| Item | Deployed setup |
| --- | --- |
| Host CPU | Hygon C86-4G (`x86_64`) |
| AI accelerator | Houmo XH2/M50 IPU |
| Vendor stack | Dadao V1.4.0, `tcim`, `tcim_lite`, `xhquant` |
| Robot | SO101 follower arm with Feetech motors |
| Sensors | Top camera + side camera |
| Policy | LeRobot PI0.5, 6-DoF SO101 action space |

## Repository Layout

```text
.
├── config/pi0/llm/                         # SO101 Gemma-2B/action-expert export configs
├── lerobot/                                # Local LeRobot source used by deployment
├── xh_model_zoo/                           # Houmo model-zoo code used by conversion
├── build_vision_so101.py                   # Build vision HMM
├── build_gemma_2b_prefill_so101.py         # Build Gemma-2B prefill HMM
├── build_gemma_expert_300m_decode_so101.py # Build action-expert decode HMM
├── build_pi05_action_time_linear_so101.py  # Build action/time projection HMMs
├── calibrate_so101.py                      # SO101 calibration helper
├── modeling_pi05_hm.py                     # TCIM-backed PI0.5 runtime
├── pi05_export_vision_xh2a_so101.py        # Vision export
├── pi05_export_llm_xh2a_libero.py          # Gemma-2B export, reused by SO101 configs
├── pi05_export_experts_xh2a_libero.py      # Action-expert export, reused by SO101 configs
├── pi05_export_other_xh2a_libero.py        # Action/time projection export
├── run_so101_hm.py                         # Real-robot inference entry point
├── so101_patch_model_config.py             # Patch checkpoint/config for export or inference
├── so101_export_quant.sh                   # One-command W8A8 export wrapper
└── so101_build_hmm.sh                      # One-command M50 HMM build wrapper
```

The runtime and conversion products are intentionally excluded from Git:
`models/`, `outputs/`, `log/`, datasets, calibration caches, ONNX/HMONNX/HMM
files, PyTorch checkpoints, and local backup files.

This repository keeps the deployment sources directly at the repository root:
`lerobot/` and `xh_model_zoo/` are first-level directories.

## Prerequisites

Start from the Houmo conversion/deployment environment that already contains
the matching vendor packages. Do not replace the vendor PyTorch, TCIM, or
XHQuant packages with ordinary PyPI wheels unless you are rebuilding the entire
stack.

Expected external model layout:

```text
models/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── policy_preprocessor.json
│   └── policy_postprocessor.json
├── paligemma-3b-pt-224/
└── pi05_base/
```

Create a log directory before running commands that use `tee`:

```bash
mkdir -p log
```

## End-to-End Usage

For a clean SO101 checkpoint in `models/pretrained_model`, the usual flow is:

```bash
python3 so101_patch_model_config.py \
  --model-dir models/pretrained_model \
  --mode export \
  --n-action-steps 30

QUANT_TYPE=w8a8_sefp bash so101_export_quant.sh models/pretrained_model \
  2>&1 | tee log/so101_export_w8a8.log

bash so101_build_hmm.sh \
  2>&1 | tee log/so101_build_w8a8.log

python3 so101_patch_model_config.py \
  --model-dir models/pretrained_model \
  --mode infer \
  --n-action-steps 30
```

After these steps, `run_so101_hm.py` uses the checkpoint metadata from
`models/pretrained_model` and the compiled Houmo runtime artifacts under
`outputs/xh2/`.

## Checkpoint Preparation

Before W8A8 export, patch the trained checkpoint into LeRobot PI0.5 export mode:

```bash
python3 so101_patch_model_config.py \
  --model-dir models/pretrained_model \
  --mode export \
  --n-action-steps 30
```

This updates lightweight JSON/config files only. It does not change model
weights. In export mode, the policy `type` is set to `pi05`, devices are patched
to CPU where needed, and the SO101 LLM config files are aligned to the selected
checkpoint path.

## W8A8 Export

Run the one-command quantized export wrapper:

```bash
QUANT_TYPE=w8a8_sefp bash so101_export_quant.sh models/pretrained_model \
  2>&1 | tee log/so101_export_w8a8.log
```

The wrapper performs the SO101 export sequence:

```text
pi05_export_vision_xh2a_so101.py
pi05_export_llm_xh2a_libero.py
pi05_export_experts_xh2a_libero.py
pi05_export_other_xh2a_libero.py
```

The `*_libero.py` names are historical filenames from the Houmo example stack;
with the SO101 config files, they are used for the SO101 Gemma/action-expert and
action/time module export path.

Export outputs are written under `outputs/xh2/`, and step logs are also written
under `outputs/logs/`.

## HMM Build

Compile the exported modules into M50 runtime artifacts:

```bash
bash so101_build_hmm.sh \
  2>&1 | tee log/so101_build_w8a8.log
```

The build wrapper runs:

```text
build_vision_so101.py
build_gemma_2b_prefill_so101.py
build_gemma_expert_300m_decode_so101.py
build_pi05_action_time_linear_so101.py
```

It also creates the runtime filenames expected by `modeling_pi05_hm.py`:

```text
outputs/xh2/siglip.hmm
outputs/xh2/gemma_2b_prefill.hmm
outputs/xh2/gemma_expert_300m_decode.hmm
outputs/xh2/action_in_proj.hmm
outputs/xh2/action_out_proj.hmm
outputs/xh2/time_mlp.hmm
outputs/xh2/embedding.pt
```

Do not mix HMM files, `embedding.pt`, or checkpoint metadata from different
training checkpoints.

## Runtime Config

After HMM build and before deployment inference, patch the same checkpoint into
Houmo runtime mode:

```bash
python3 so101_patch_model_config.py \
  --model-dir models/pretrained_model \
  --mode infer \
  --n-action-steps 30
```

In inference mode, the policy `type` is set to `hm_pi05`, so
`run_so101_hm.py` loads the TCIM-backed policy in `modeling_pi05_hm.py`.

## SO101 Inference

Start with a dry run by using the same command below without `--send-action`.
In dry-run mode, the script reads observations and predicts actions, but does
not command the motors.

```bash
python3 -u run_so101_hm.py \
  --model-path models/pretrained_model \
  --task "pick up the block and place it into the yellow box" \
  --port /dev/ttyACM2 \
  --robot-id so101_follower \
  --use-degrees \
  --max-relative-target 10.0 \
  --camera-type opencv \
  --top-camera 0 \
  --side-camera 2 \
  --top-camera-fourcc MJPG \
  --side-camera-fourcc MJPG \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --fps 10 \
  --duration-s 60 \
  --send-action
```

Adjust camera indices, serial port, task text, and safety limits to match your
actual robot and training setup.

### Runtime Options

| Option | Purpose |
| --- | --- |
| `--model-path` | Checkpoint directory used for config and processors |
| `--robot-id` | Robot calibration/runtime identifier |
| `--port` | Feetech motor serial port |
| `--camera-type` | Camera backend, for example `opencv` |
| `--top-camera`, `--side-camera` | Top and side camera device indices/paths |
| `--top-camera-fourcc`, `--side-camera-fourcc` | Camera pixel format, for example `MJPG` |
| `--camera-width`, `--camera-height` | Camera capture resolution |
| `--camera-fps`, `--fps` | Camera capture rate and policy loop rate |
| `--task` | Instruction text; use the training task wording |
| `--use-degrees` | Use degree-based arm joints and percentage-style gripper values |
| `--max-relative-target` | Per-step motor target safety limit |
| `--send-action` | Enable actual motor commands |

## Community

Join the RadixRootMind China developer WeChat group:

<img src="https://github.com/RadixRootMind/RadixQuantVLA/raw/main/assets/RadixRootMind_group.png" alt="RadixRootMind China developer WeChat group" width="320">

## Acknowledgements and License

This integration builds on [LeRobot](https://github.com/huggingface/lerobot),
the PI0.5/OpenPI model family, and the Houmo Dadao/XH2 deployment toolchain.
LeRobot's license is preserved under `lerobot/LICENSE`.

See `NOTICE.md` for third-party boundaries and distribution details.
