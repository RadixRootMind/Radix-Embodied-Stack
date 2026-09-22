<<<<<<< HEAD
# PI0.5 on Houmo M50 for SO101

**From a LeRobot PI0.5 checkpoint to W8A8 HMONNX, M50 HMM, and SO101 closed-loop inference**

[![Related project: RadixQuantVLA Stars](https://img.shields.io/github/stars/RadixRootMind/RadixQuantVLA?style=social&label=RadixQuantVLA%20Stars)](https://github.com/RadixRootMind/RadixQuantVLA) [![LeRobot](https://img.shields.io/badge/Based%20on-LeRobot-blue)](https://github.com/huggingface/lerobot) [![Hardware](https://img.shields.io/badge/Target-Houmo%20M50-green)](#prerequisites) [![CPU](https://img.shields.io/badge/Host-Hygon%20C86--4G-red)](#prerequisites) [![Robot](https://img.shields.io/badge/Robot-SO101-orange)](#so101-inference)

[Overview](#overview) | [Platforms](#supported-platforms) | [Architecture](docs/ARCHITECTURE.md) | [Prerequisites](#prerequisites) | [Export](#export-and-quantization) | [Compile](#compile-for-m50) | [Inference](#so101-inference) | [License](#acknowledgements-and-license)

> **Status:** This code was organized from an SO101 real-robot deployment.
> Hardware results are not reported in this repository.

## Overview

This project connects a fine-tuned LeRobot PI0.5 policy to a Houmo XH2/M50
accelerator and an SO101 follower arm on a Hygon C86-4G host. It covers model
preparation, module export, Houmo PTQ, HMM compilation, camera/state acquisition,
and motor control.
The original trained checkpoint is still needed for configuration, tokenizer,
and pre/postprocessing; neural-network inference uses the compiled HMM files.

**Key Features**

- 🔧 **Model conversion:** split a LeRobot PI0.5 checkpoint into deployable modules.
- ⚡ **M50 inference:** export HMONNX, compile HMM, and run with TCIM.
- 📷 **Two-camera input:** combine top/side images with SO101 joint state and task text.
- 🦾 **Robot control:** turn predicted actions into calibrated Feetech motor targets.

| Component | Role | Implementation |
| --- | --- | --- |
| LeRobot PI0.5 | Policy configuration and pre/postprocessing | `third_party/lerobot/` |
| Houmo export | Split modules and produce HMONNX | `export/`, `config/` |
| M50 compiler | Produce runtime HMM binaries | `build/` |
| TCIM runtime | Run the compiled modules | `modeling_pi05_hm.py` |
| SO101 loop | Read cameras/joints and send actions | `run_so101_hm.py` |

The path from sensors to motors is documented in [Architecture](docs/ARCHITECTURE.md).
This is a Houmo deployment integration, **not** an implementation of the
[QuantVLA](https://github.com/AIoT-MLSys-Lab/QuantVLA) quantization algorithm.

## Supported Platforms

**Host CPU:** Hygon C86-4G (`x86_64`)

**AI accelerator:** Houmo XH2/M50 (IPU), using the Dadao V1.4.0 toolchain and TCIM runtime

**Robot:** SO101 follower robotic arm with Feetech motors and top/side cameras

**Model/software:** LeRobot PI0.5, Houmo PTQ/HMONNX/HMM export and inference

## Repository Layout

```text
.
├── build/                  # HMONNX -> M50 HMM
├── config/pi0/llm/         # Gemma-2B and action-expert graph settings
├── docs/                   # Architecture and data contracts
├── export/                 # PI0.5 -> ONNX/HMONNX
├── third_party/
│   ├── lerobot/            # Tested, modified LeRobot 0.4.4 source
│   └── xh_model_zoo/      # Add locally from Houmo examples; not in Git
├── calibrate_so101.py     # SO101 calibration
├── modeling_pi05_hm.py    # TCIM-backed PI0.5 runtime
├── prepare_model.py       # Export/runtime checkpoint configuration
└── run_so101_hm.py        # Real-robot inference entry point
```

Model weights, datasets, LIBERO assets, calibration data, logs, ONNX/HMONNX,
HMM files, and vendor SDK binaries are intentionally excluded by `.gitignore`.

## Prerequisites

| Item | Requirement |
| --- | --- |
| Deployed host CPU | Hygon C86-4G (x86_64) |
| Accelerator | Houmo XH2/M50 with a working driver and firmware |
| Vendor stack | Compatible Dadao V1.4.0 conversion/deployment environment, `tcim`, `tcim_lite`, `xhquant` |
| Model zoo | Matching `xh_model_zoo` from the Houmo examples package |
| Checkpoint | Fine-tuned LeRobot PI0.5 with two image inputs and six SO101 actions |
| Robot | SO101 follower, Feetech USB serial adapter, and top/side cameras |

The deployed combination is **Hygon C86-4G + Houmo M50 + SO101**.

Start from the vendor-provided environment. **Do not replace its PyTorch, TCIM,
or XHQuant builds with ordinary PyPI packages.** Additional Python dependencies
used during the original export are recorded in
[`requirements.pi05-v1.4.0.txt`](requirements.pi05-v1.4.0.txt); install them in
an isolated copy of the vendor environment and resolve any vendor version
constraints before running conversion.

Place the vendor model zoo at `third_party/xh_model_zoo/` as described in
[`third_party/README.md`](third_party/README.md). Copy `.env.example` to a
local environment file and adjust paths for your container; never commit the
result or any access token.

## Checkpoint Preparation

Arrange the external files as follows:

```text
models/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── policy_preprocessor.json
│   └── policy_postprocessor.json
└── paligemma-3b-pt-224/
```

Run all commands below from the repository root inside the Houmo conversion
container. Before export, set the policy type to `pi05` and the device to CPU:

```bash
python3 prepare_model.py models/pretrained_model --mode export
```

The helper updates the model's `config.json` and preprocessor JSON. Verify that
the checkpoint's image keys, six-action order, normalization statistics, and
tokenizer match the training dataset before continuing.

## Export and Quantization

Export the vision tower, Gemma-2B prefill, Gemma action expert, and action/time
projection modules:

```bash
python3 export/vision.py --model_path ./models/pretrained_model
python3 export/gemma_2b.py --config ./config/pi0/llm/pi05_gemma_2b_xh2a_2k_so101_mask.py
python3 export/gemma_expert.py --config ./config/pi0/llm/pi05_gemma_expert_300m_xh2a_2k_so101_mask.py
python3 export/action_heads.py --model_path ./models/pretrained_model
```

The vision and action-head scripts request Houmo `w8a8h1_sefp`; the Gemma
components use the model-zoo PTQ path. These scripts are CPU-side conversion
steps: a zero IPU utilization reading during export does not imply that the
eventual HMM inference will run on CPU. Intermediate files are written beneath
`artifacts/xh2/` and are not tracked by Git.

## Compile for M50

```bash
python3 build/build_vision_so101.py
python3 build/build_gemma_2b_prefill_so101.py
python3 build/build_gemma_expert_300m_decode_so101.py
python3 build/build_pi05_action_time_linear_so101.py
```

The runtime expects these seven files under `artifacts/xh2/`:

```text
siglip.hmm
gemma_2b_prefill.hmm
gemma_expert_300m_decode.hmm
action_in_proj.hmm
action_out_proj.hmm
time_mlp.hmm
embedding.pt
```

Set `HOUMO_ARTIFACT_DIR` when the compiled files live elsewhere. The runtime
checks for all seven files before initializing TCIM. Do not mix artifacts from
different checkpoints.

## SO101 Inference

Calibrate the robot using the actual USB serial port and keep its calibration
file outside Git:

```bash
python3 calibrate_so101.py --port /dev/ttyACM0 --robot-id so101_follower
python3 prepare_model.py models/pretrained_model --mode runtime
```

Start with a read-and-predict run. **Without `--send-action`, the script does
not send policy actions to the motors.** Confirm camera identity and supported
resolution/FPS on your own machine; Linux `/dev/video*` indices can change.

```bash
python3 -u run_so101_hm.py \
  --model-path ./models/pretrained_model \
  --duration-s 10 --fps 10 \
  --port /dev/ttyACM0 \
  --top-camera 6 --side-camera 8 \
  --camera-fps 10 \
  --no-calibrate --use-degrees --no-clip-action-range \
  --task "夹起方块并放入白色盒子里，放入成功拍一下黄色按钮"
```

After checking observations, action values, calibration, and the workspace,
repeat the command with `--send-action`. Keep the first powered run short and
retain a conservative `--max-relative-target`. The example task text and camera
indices describe one setup, not universal defaults. `--use-degrees` reflects a
dataset whose first five joints are angles and whose gripper is a percentage.

### Key Runtime Options

| Option | Purpose |
| --- | --- |
| `--model-path` | Checkpoint directory used for policy metadata and processors |
| `--top-camera`, `--side-camera` | Top and side camera device indices/paths |
| `--camera-fps`, `--fps` | Camera capture rate and action-loop target rate |
| `--task` | Instruction text; use the training task wording |
| `--use-degrees` | Degree-based arm joints, percentage-based gripper |
| `--max-relative-target` | Per-step motor target limit |
| `--no-calibrate` | Reuse the existing calibration file |
| `--send-action` | Enable actual motor commands; omit for read-and-predict |

## Verification and Limitations

- This release is source-only. It cannot run without the external checkpoint,
  vendor software, compiled artifacts, calibration, and hardware.
- No benchmark success rate, latency, or accuracy claim is made for this release.
- A checkpoint trained with different camera keys, action units, or task text
  requires corresponding preprocessing changes; the HMM alone is insufficient.

## Acknowledgements and License

The integration builds on [LeRobot](https://github.com/huggingface/lerobot) and
the PI0.5/OpenPI model family. LeRobot's Apache-2.0 license is preserved under
`third_party/lerobot/LICENSE`. [QuantVLA](https://github.com/AIoT-MLSys-Lab/QuantVLA)
inspired the organization of this README only; its PTQ implementation is not
included.

See [NOTICE.md](NOTICE.md) for third-party boundaries and distribution details.
=======
# Radix Embodied Stack

Universal infrastructure for embodied AI: model quantization, inference runtime, and robot control for domestic chips.

## Overview

Radix Embodied Stack enables **cross-model, cross-chip, cross-platform** deployment of embodied AI models (VLA, VLM+Policy, Diffusion Policy, World-Action Models) on domestic computing platforms.

**Key Features:**
- 🔧 Model quantization and optimization
- ⚡ Inference runtime with hardware adaptation  
- 🤖 Multi-robot control interface
- 🎯 Support for domestic chips

## Supported Platforms

**Domestic chips:**
TPU/GPU/NPU/IPU

**Robots:**
- Humanoid robots 
- Quadruped robots
- Mobile manipulators
- Robotic arms
- Dexterous hands 

## Architecture

```
Model Layer     →  VLA / VLM+Policy / Diffusion Policy
    ↓
Infra Layer     →  Quantization / Runtime / Control
    ↓
Hardware Layer  →  Domestic Chips / Robot Platforms
```

## Roadmap

- [ ] Model quantization toolkit
- [ ] Inference runtime for domestic chips
- [ ] Robot control adapters
- [ ] Benchmarks and performance profiles
- [ ] Documentation and tutorials

## Contributing

Contributions welcome! This project is part of the [RadixRootMind](https://github.com/RadixRootMind) open-source community.

## License

TBD

---

**Project Status:** 🚧 Under active development
>>>>>>> origin/main
