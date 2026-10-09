#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "lerobot" / "src"))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.utils import build_dataset_frame
from lerobot.motors import MotorCalibration
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action

from modeling_pi05_hm import ARTIFACT_DIR, HMPI05Policy


_MOTOR_CALIBRATION_TYPES = {
    "id": int,
    "drive_mode": int,
    "homing_offset": int,
    "range_min": int,
    "range_max": int,
}
MotorCalibration.__annotations__ = _MOTOR_CALIBRATION_TYPES
for _field_name, _field_type in _MOTOR_CALIBRATION_TYPES.items():
    MotorCalibration.__dataclass_fields__[_field_name].type = _field_type


MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Houmo-compiled PI0.5 policy on a SO101 follower arm.")
    parser.add_argument("--model-path", default="./models/pretrained_model")
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=None, help="Run for this many seconds; overrides --steps.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--disable-torque-on-disconnect", action="store_true")
    parser.add_argument("--motor-read-retries", type=int, default=5)
    parser.add_argument("--motor-read-retry-sleep-s", type=float, default=0.02)
    parser.add_argument(
        "--clip-action-range",
        action="store_true",
        help="Optionally clip absolute action values before send_action; disabled by default to match LeRobot rollout.",
    )
    parser.add_argument("--no-clip-action-range", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-degrees", action="store_true", help="Use joint angles in degrees, matching the training dataset.")
    parser.add_argument(
        "--max-relative-target",
        default="10.0",
        help="Maximum per-step relative motor target. Use 'none' to disable this safety clamp.",
    )
    parser.add_argument("--camera-type", choices=["opencv", "realsense"], default="opencv")
    parser.add_argument("--top-camera", required=True)
    parser.add_argument("--side-camera", required=True)
    parser.add_argument("--top-camera-fourcc", default="MJPG")
    parser.add_argument("--side-camera-fourcc", default="MJPG")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--top-camera-width", type=int, default=None)
    parser.add_argument("--top-camera-height", type=int, default=None)
    parser.add_argument("--side-camera-width", type=int, default=None)
    parser.add_argument("--side-camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Use dummy observations and do not connect the robot.")
    parser.add_argument("--send-action", action="store_true", help="Actually send predicted actions to the SO101.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--debug-observation", action="store_true", help="Print real observation/image stats for the first few steps.")
    parser.add_argument("--debug-observation-steps", type=int, default=3)
    parser.add_argument("--debug-image-dir", default=None, help="Optional directory to save captured top/side frames for debugging.")
    return parser.parse_args()


def parse_max_relative_target(value: str) -> float | None:
    if value.lower() in {"none", "null", "off", "false", "no"}:
        return None
    return float(value)


def camera_config(kind: str, camera_id: str, width: int, height: int, fps: int, fourcc: str | None = None):
    if kind == "opencv":
        index_or_path = int(camera_id) if camera_id.isdigit() else camera_id
        return OpenCVCameraConfig(index_or_path=index_or_path, width=width, height=height, fps=fps, fourcc=fourcc)
    return RealSenseCameraConfig(serial_number_or_name=camera_id, width=width, height=height, fps=fps)


def dataset_features(top_height: int, top_width: int, side_height: int, side_width: int) -> dict:
    return {
        "observation.images.top": {
            "dtype": "image",
            "shape": (top_height, top_width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.side": {
            "dtype": "image",
            "shape": (side_height, side_width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTOR_NAMES),),
            "names": MOTOR_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTOR_NAMES),),
            "names": MOTOR_NAMES,
        },
    }


def check_model_config(model_path: Path) -> None:
    cfg_path = model_path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("type") != "hm_pi05":
        raise RuntimeError(f"{cfg_path} has type={cfg.get('type')!r}; set it to 'hm_pi05' before Houmo inference.")
    shape = cfg.get("output_features", {}).get("action", {}).get("shape")
    # pi05 内部动作空间是 [32], 后处理 postprocessor 会自动转回 6/7 维关节动作
    if shape not in ([6], [7], [32]):
        raise RuntimeError(f"{cfg_path} action shape={shape!r}; expected [6]/[7]/[32] for SO101.")
        raise RuntimeError(f"{cfg_path} action shape={shape!r}; expected [6]/[7]/[32] for SO101.")


def load_policy(model_path: Path):
    required_artifacts = (
        "siglip.hmm",
        "gemma_2b_prefill.hmm",
        "gemma_expert_300m_decode.hmm",
        "action_in_proj.hmm",
        "action_out_proj.hmm",
        "time_mlp.hmm",
        "embedding.pt",
    )
    missing = [name for name in required_artifacts if not (ARTIFACT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Houmo runtime artifacts under {ARTIFACT_DIR}: {', '.join(missing)}"
        )
    policy = HMPI05Policy.from_pretrained(str(model_path))
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(model_path),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()
    preprocess.reset()
    postprocess.reset()
    return policy, preprocess, postprocess


def make_dummy_observation(top_height: int, top_width: int, side_height: int, side_width: int) -> dict:
    obs = {name: np.float32(0.0) for name in MOTOR_NAMES}
    obs["top"] = np.zeros((top_height, top_width, 3), dtype=np.uint8)
    obs["side"] = np.zeros((side_height, side_width, 3), dtype=np.uint8)
    return obs




def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def debug_observation(step: int, obs: dict, image_dir: str | None = None) -> None:
    keys = list(obs.keys())
    print(f"[debug obs {step:04d}] keys={keys}", flush=True)
    for cam_key in ("top", "side"):
        if cam_key not in obs:
            print(f"[debug obs {step:04d}] missing camera key: {cam_key}", flush=True)
            continue
        arr = np.asarray(obs[cam_key])
        if arr.size:
            stats = f"min={float(np.nanmin(arr)):.3f} max={float(np.nanmax(arr)):.3f} mean={float(np.nanmean(arr)):.3f}"
        else:
            stats = "empty"
        print(f"[debug obs {step:04d}] {cam_key}: shape={arr.shape} dtype={arr.dtype} {stats}", flush=True)
        if image_dir is not None:
            out_dir = Path(image_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{step:04d}_{cam_key}.png"
            try:
                from PIL import Image

                Image.fromarray(_image_to_uint8(arr)).save(out_path)
                print(f"[debug obs {step:04d}] saved {cam_key} frame: {out_path}", flush=True)
            except Exception as exc:
                logging.warning("Failed to save debug image %s: %s", out_path, exc, exc_info=True)


def make_robot(args: argparse.Namespace) -> SO101Follower:
    top_width = args.top_camera_width or args.camera_width
    top_height = args.top_camera_height or args.camera_height
    side_width = args.side_camera_width or args.camera_width
    side_height = args.side_camera_height or args.camera_height
    cameras = {
        "top": camera_config(args.camera_type, args.top_camera, top_width, top_height, args.camera_fps, args.top_camera_fourcc),
        "side": camera_config(args.camera_type, args.side_camera, side_width, side_height, args.camera_fps, args.side_camera_fourcc),
    }
    cfg_kwargs = {
        "id": args.robot_id,
        "port": args.port,
        "cameras": cameras,
        "max_relative_target": parse_max_relative_target(args.max_relative_target),
        "disable_torque_on_disconnect": args.disable_torque_on_disconnect,
        "use_degrees": args.use_degrees,
    }
    if args.calibration_dir:
        cfg_kwargs["calibration_dir"] = args.calibration_dir
    return SO101Follower(SO101FollowerConfig(**cfg_kwargs))


def run_once(obs: dict, features: dict, policy, preprocess, postprocess, task: str, robot_type: str) -> dict:
    frame = build_dataset_frame(features, obs, prefix=OBS_STR)
    action_tensor = predict_action(
        observation=frame,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=preprocess,
        postprocessor=postprocess,
        use_amp=False,
        task=task,
        robot_type=robot_type,
    )
    return make_robot_action(action_tensor, features)


def clip_action_range(action: dict) -> dict:
    clipped = {}
    changed = {}
    for key, value in action.items():
        lo, hi = (0.0, 100.0) if key == "gripper.pos" else (-100.0, 100.0)
        safe_value = min(max(float(value), lo), hi)
        clipped[key] = safe_value
        if safe_value != float(value):
            changed[key] = {"original": float(value), "clipped": safe_value}
    if changed:
        logging.warning("Clipped action to SO101 normalized range: %s", changed)
    return clipped


def get_observation_with_retries(robot: SO101Follower, args: argparse.Namespace) -> dict:
    last_exc: Exception | None = None
    for attempt in range(args.motor_read_retries + 1):
        try:
            return robot.get_observation()
        except ConnectionError as exc:
            last_exc = exc
            if attempt >= args.motor_read_retries:
                break
            logging.warning(
                "SO101 state read failed (%s/%s): %s",
                attempt + 1,
                args.motor_read_retries + 1,
                exc,
            )
            time.sleep(args.motor_read_retry_sleep_s)
    raise last_exc if last_exc is not None else RuntimeError("SO101 state read failed")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    model_path = Path(args.model_path)
    check_model_config(model_path)
    top_width = args.top_camera_width or args.camera_width
    top_height = args.top_camera_height or args.camera_height
    side_width = args.side_camera_width or args.camera_width
    side_height = args.side_camera_height or args.camera_height
    features = dataset_features(top_height, top_width, side_height, side_width)
    policy, preprocess, postprocess = load_policy(model_path)

    robot = None
    robot_type = "so101_follower"
    if not args.dry_run:
        robot = make_robot(args)
        robot.connect(calibrate=not args.no_calibrate)
        robot_type = robot.robot_type

    try:
        max_steps: int | None = args.steps
        end_time: float | None = None
        if args.duration_s is not None:
            end_time = time.perf_counter() + args.duration_s
            max_steps = None

        step = 0
        while max_steps is None or step < max_steps:
            if end_time is not None and time.perf_counter() >= end_time:
                break
            start = time.perf_counter()
            obs = (
                make_dummy_observation(top_height, top_width, side_height, side_width)
                if args.dry_run
                else get_observation_with_retries(robot, args)
            )
            if args.debug_observation and step < args.debug_observation_steps:
                debug_observation(step, obs, args.debug_image_dir)
            action = run_once(obs, features, policy, preprocess, postprocess, args.task, robot_type)
            if args.clip_action_range and not args.no_clip_action_range:
                action = clip_action_range(action)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"[{step:04d}] {elapsed_ms:.1f} ms action={action}", flush=True)
            if args.send_action and robot is not None:
                sent = robot.send_action(action)
                print(f"[{step:04d}] sent={sent}", flush=True)
            if args.fps > 0:
                time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - start)))
            step += 1
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                logging.warning("Ignoring SO101 disconnect error: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
