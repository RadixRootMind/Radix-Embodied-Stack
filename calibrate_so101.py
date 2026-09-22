#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "lerobot" / "src"))

from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a SO101 follower arm without draccus CLI parsing.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="so101_follower")
    parser.add_argument("--calibration-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "port": args.port,
        "id": args.robot_id,
    }
    if args.calibration_dir:
        kwargs["calibration_dir"] = args.calibration_dir

    robot = SO101Follower(SO101FollowerConfig(**kwargs))
    robot.connect(calibrate=False)
    try:
        robot.calibrate()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
