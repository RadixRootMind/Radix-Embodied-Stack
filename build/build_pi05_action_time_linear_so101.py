import json
import tcim
import logging
import os
import psutil
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

project_root = Path(__file__).resolve().parents[1]
hmonnx_dir = str(project_root / "artifacts/xh2/hmonnx")
output_dir = str(project_root / "artifacts/xh2")

model_name = "action_in_proj"
model_path = os.path.join(hmonnx_dir, "action_in_proj.onnx")


tcim.build_from_hmonnx(
    model_path,
    output_name=model_name,
    ncore=2,
    target="xh2",
    output_dir=output_dir,
    work_dir=os.path.join(output_dir, "tcim", model_name),
    j=psutil.cpu_count(logical=False),
    llm_opt=False,
    skip_mlir_compile=False,
)

model_name = "action_out_proj"
model_path = os.path.join(hmonnx_dir, "action_out_proj.onnx")

tcim.build_from_hmonnx(
    model_path,
    output_name=model_name,
    ncore=2,
    target="xh2",
    output_dir=output_dir,
    work_dir=os.path.join(output_dir, "tcim", model_name),
    j=psutil.cpu_count(logical=False),
    llm_opt=False,
    skip_mlir_compile=False,
)

model_name = "time_mlp"
model_path = os.path.join(hmonnx_dir, "time_mlp.onnx")

tcim.build_from_hmonnx(
    model_path,
    output_name=model_name,
    ncore=2,
    target="xh2",
    output_dir=output_dir,
    work_dir=os.path.join(output_dir, "tcim", model_name),
    j=psutil.cpu_count(logical=False),
    llm_opt=False,
    skip_mlir_compile=False,
)
