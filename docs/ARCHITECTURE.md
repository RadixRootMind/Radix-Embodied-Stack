# Architecture

## Runtime ownership

`run_so101_hm.py` owns hardware discovery, camera configuration, serial-bus
connection, control-loop timing, safety clipping, and conversion between LeRobot
frames and SO101 action dictionaries.

`modeling_pi05_hm.py` owns the PI0.5 algorithm and TCIM runtime. It loads seven
artifacts: SigLIP, Gemma-2B prefill, Gemma expert decode, three action/time
projection modules, and the token embedding tensor.

The bundled LeRobot tree owns camera drivers, Feetech motor I/O, calibration,
normalization, policy configuration, task tokenization, and action postprocessing.

Houmo `tcim_lite` owns M50 device memory, module loading, tensor transfer, and
execution. It is an external binary dependency.

## Export flow

```text
trained PI0.5 checkpoint (type=pi05, CPU mode)
  -> split PyTorch modules
  -> ONNX export and ONNX Runtime numerical check
  -> xhquant W8A8 HMONNX
  -> tcim.build_from_hmonnx(target=xh2, ncore=2)
  -> seven runtime artifacts
```

The checkpoint remains necessary at runtime for configuration, normalization,
tokenization, and LeRobot pre/post-processors. Neural network execution uses the
compiled HMM files, not `model.safetensors`.

## Observation contract

- `observation.images.top`: RGB top camera
- `observation.images.side`: RGB side camera
- `observation.state`: six values in training order
- `task`: exact language instruction used by the dataset

The motor order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist
roll, and gripper. The deployed dataset used degrees for the first five joints
and percentage for the gripper, so runtime uses `--use-degrees`.

## Action contract

PI0.5 predicts a chunk of actions. LeRobot postprocessing restores dataset units,
then `make_robot_action` maps the six values to named SO101 targets. The robot
bus applies calibration and optional per-step relative-target limits before
writing Feetech goal positions.

## Local LeRobot changes

The included source contains the compatibility changes required by the deployed
environment, notably:

- `hm_pi05` configuration and policy registration
- tolerant checkpoint `type` removal during config loading
- SO101 `use_degrees` support
- the PI0.5/OpenPI-compatible policy implementation used by the checkpoint

Keep this tree pinned until those changes are rebased and tested against a newer
upstream LeRobot release.
