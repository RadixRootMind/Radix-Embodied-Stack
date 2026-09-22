quant_config = dict(inputs=dict())
target_device = "XH2a"
frontend_type = "TorchFX"
hf_model_dir = "./models/pretrained_model"
config_dir = "./models/paligemma-3b-pt-224"

quant_config = dict()
model = dict(
    type="XHGemmaLLMModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=1024,
        input_sequence_length=968,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        # only_first_block=True,
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "attention_mask",
        ],
        output_names=["last_hidden_state"],
    ),
)
