import json
import tcim
import logging
import os
import psutil

logging.basicConfig(level=logging.DEBUG)

model_name = "gemma_expert_300m_decode_so101"
model_path = "outputs/xh2/hmonnx/pi05_gemma_expert_300m_xh2a_2k_so101_mask/decode_onnx/pi05_gemma_expert_300m_xh2a_2k_so101_mask_decode.onnx"
output_dir = "outputs/xh2/"

kwargs = dict()
custom_msg = dict()
flash_attention = 0
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
    llm_opt=True,
    skip_mlir_compile=False,
    enable_common_subgraph=False,
    subgraph_repeat_hint=-18,
    **kwargs,
)
