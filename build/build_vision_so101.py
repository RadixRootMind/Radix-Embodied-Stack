import json
import tcim
import logging
import os
import psutil
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

project_root = Path(__file__).resolve().parents[1]
model_name = "siglip"
model_path = str(project_root / "artifacts/xh2/hmonnx/pi05_vision_xh2a_so101/vision.onnx")
output_dir = str(project_root / "artifacts/xh2")

kwargs = dict()
custom_msg = dict()
flash_attention = 2
kwargs["flash_attention"] = flash_attention
custom_msg["flash_attention"] = flash_attention
kwargs["custom_msg"] = json.dumps(custom_msg, ensure_ascii=False)
print(f"\n===> {model_name} build start...")

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
    enable_common_subgraph=False,
    subgraph_repeat_hint=-27,
    **kwargs,
)
