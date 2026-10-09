import torch
from safetensors.torch import load_file

model_dir = "/workspace/pi0.5/models/pi05_merged_40000"

# 用正确的类名
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

config = PI05Config()
config.dtype = "bfloat16"

model = PI05Pytorch(config)

sd = load_file(f"{model_dir}/model.safetensors")
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
if missing:
        print("Missing 示例:", missing[:5])
        if unexpected:
                print("Unexpected 示例:", unexpected[:5])

                print(f"模型加载成功, 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
