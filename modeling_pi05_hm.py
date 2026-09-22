import os
import builtins
import time
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict
from typing_extensions import Unpack

import tcim_lite as tcim
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, LongTensor, FloatTensor
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.pi05.configuration_pi05 import HMPI05Config
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

PRINT_DEBUG = os.environ.get("DEBUG", "0")
assert PRINT_DEBUG in ["0", "1"], "DEBUG must be 0 or 1"
PRINT_DEBUG = PRINT_DEBUG == "1"
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("HOUMO_ARTIFACT_DIR", PROJECT_ROOT / "artifacts" / "xh2"))


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(
    pad_masks, att_masks
):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(
            0, 2, 3, 1
        )  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma model with action expert for PI05."""

    def __init__(self):
        super().__init__()
        vision_model_path = ARTIFACT_DIR / "siglip.hmm"
        gemma_2b_prefill_model_path = ARTIFACT_DIR / "gemma_2b_prefill.hmm"
        gemma_expert_300m_decode_model_path = (
            ARTIFACT_DIR / "gemma_expert_300m_decode.hmm"
        )
        embedding_path = ARTIFACT_DIR / "embedding.pt"
        wm = tcim.runtime.WeightManager(device=0)
        option0 = tcim.runtime.Option(wm)
        option1 = tcim.runtime.Option(wm)
        option2 = tcim.runtime.Option(wm)
        self.paligemma = tcim.runtime.load(str(vision_model_path), option0)
        self.gemma_2b = tcim.runtime.load(str(gemma_2b_prefill_model_path), option1)
        # show input info
        for idx in range(self.gemma_2b.get_num_inputs()):
            name = self.gemma_2b.get_input_name(idx)
            info = self.gemma_2b.get_input_info(name)
            if name.startswith("model_layer"):
                continue
            print(
                f"[Pi05][gemma-2b] input_info[{name}] shape = {info.shape}, dtype = {info.dtype}, format = {info.format}, stride = {info.stride}"
            )
        dummy_tensor_names = self.get_dummy_tensor_names()
        self.gemma_300m_expert = tcim.runtime.load(
            str(gemma_expert_300m_decode_model_path), option2
        )
        # pass kvcache
        for name in dummy_tensor_names:
            self.gemma_300m_expert.set_input(name, self.gemma_2b.get_dev_input(name))

        self.gemma_expert_token = None
        weight = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embed_tokens = nn.Embedding(
            num_embeddings=weight["weight"].shape[0],
            embedding_dim=weight["weight"].shape[1],
        )
        self.embed_tokens.load_state_dict(weight)
        self.valid_kvcache_length = 0

    def get_dummy_tensor_names(self):
        dummy_tensor_names = list()
        for idx in range(self.gemma_2b.get_num_inputs()):
            name = self.gemma_2b.get_input_name(idx)
            if "model_layer" in name and "cache" in name:
                dummy_tensor_names.append(name)
        return dummy_tensor_names

    def embed_image(self, image: Tensor):
        input_name = "pixel_values"
        output_name = "image_features"
        if image.dtype != torch.float16:
            image = image.half()
        image = image.contiguous().detach().cpu().numpy()
        t0 = time.time()
        self.paligemma.set_input(input_name, image)
        t1 = time.time()
        self.paligemma.run()
        self.paligemma.sync()
        t2 = time.time()
        image_embeds = self.paligemma.get_output(output_name).numpy()  # float16
        t3 = time.time()
        if PRINT_DEBUG:
            print(
                f"[Pi05][visual] setinput: {(t1-t0)*1000:.3f}ms, infer: {(t2-t1)*1000:.3f}ms, getoutput: {(t3-t2)*1000:.3f}ms, total: {(t3-t0)*1000:.3f}ms"
            )
        return torch.from_numpy(image_embeds)

    def embed_language_tokens(self, tokens: Tensor):
        return self.embed_tokens(tokens)

    def prefill(
        self,
        attention_mask: Tensor | None = None,
        inputs_embeds: FloatTensor | None = None,
    ):
        if inputs_embeds.dtype != torch.float16:
            inputs_embeds = inputs_embeds.half()

        pad_len = 1024 - attention_mask.size(-1)
        if pad_len > 0:
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, pad_len), value=-2.3820e38
            )

        # 再在第二个维度复制到 8
        # attention_mask = attention_mask.expand(
        #     attention_mask.size(0),
        #     8,
        #     attention_mask.size(2),
        #     attention_mask.size(3),
        # )
        if attention_mask.dtype != torch.float16:
            attention_mask = attention_mask.half()

        valid_length = torch.tensor(
            [self.valid_kvcache_length], dtype=torch.int32, device=inputs_embeds.device
        )
        current_length = torch.tensor(
            [inputs_embeds.shape[1]], dtype=torch.int32, device=inputs_embeds.device
        )
        inputs_embeds = inputs_embeds.contiguous().detach().cpu().numpy()
        valid_length = valid_length.detach().cpu().numpy()
        current_length = current_length.detach().cpu().numpy()
        attention_mask = attention_mask.contiguous().detach().cpu().numpy()
        # if PRINT_DEBUG:
        #     print(
        #         f"[Pi05][gemma-2b] {"inputs_embeds"} {inputs_embeds.shape}, {inputs_embeds.dtype}, {inputs_embeds.strides}"
        #     )
        #     print(
        #         f"[Pi05][gemma-2b] {"attention_mask"} {attention_mask.shape}, {attention_mask.dtype}, {attention_mask.strides}"
        #     )
        t0 = time.time()
        self.gemma_2b.set_input("input_1", inputs_embeds)
        self.gemma_2b.set_input("valid_length", valid_length)
        self.gemma_2b.set_input("current_length", current_length)
        self.gemma_2b.set_input("attention_mask", attention_mask)
        t1 = time.time()
        self.gemma_2b.run()
        self.gemma_2b.sync()
        t2 = time.time()
        if PRINT_DEBUG:
            print(
                f"[Pi05][gemma-2b] setinput: {(t1-t0)*1000:.3f}ms, infer: {(t2-t1)*1000:.3f}ms, total: {(t2-t0)*1000:.3f}ms"
            )
        self.valid_kvcache_length = inputs_embeds.shape[1]

    def decode(
        self,
        attention_mask: Tensor,
        inputs_embeds: FloatTensor,
        adarms_cond: Tensor | None = None,
    ):
        if attention_mask.dtype != torch.float16:
            attention_mask = attention_mask.half()

        if inputs_embeds.dtype != torch.float16:
            inputs_embeds = inputs_embeds.half()

        valid_length = torch.tensor(
            [self.valid_kvcache_length], dtype=torch.int32, device=inputs_embeds.device
        )
        current_length = torch.tensor(
            [inputs_embeds.shape[1]], dtype=torch.int32, device=inputs_embeds.device
        )
        t0 = time.time()
        self.gemma_300m_expert.set_input(
            "input_1", inputs_embeds.contiguous().detach().cpu().numpy()
        )
        self.gemma_300m_expert.set_input(
            "valid_length", valid_length.detach().cpu().numpy()
        )
        self.gemma_300m_expert.set_input(
            "current_length",
            current_length.detach().cpu().numpy(),
        )
        self.gemma_300m_expert.set_input(
            "cond", adarms_cond.contiguous().detach().cpu().numpy()
        )
        self.gemma_300m_expert.set_input(
            "attention_mask", attention_mask.contiguous().detach().cpu().numpy()
        )
        t1 = time.time()
        self.gemma_300m_expert.run()
        self.gemma_300m_expert.sync()
        t2 = time.time()
        suffix_out = self.gemma_300m_expert.get_output(
            self.gemma_300m_expert.get_output_name(0)
        ).numpy()
        t3 = time.time()
        if PRINT_DEBUG:
            print(
                f"[Pi05][gemma-300m-expert] setinput: {(t1-t0)*1000:.3f}ms, infer: {(t2-t1)*1000:.3f}ms, getoutput: {(t3-t2)*1000:.3f}ms, total: {(t3-t0)*1000:.3f}ms"
            )
        return torch.from_numpy(suffix_out)


class PI05PolicyXH2a(nn.Module):
    """Core PI05 XH2a model."""

    def __init__(self, config: HMPI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        self.paligemma_with_expert = PaliGemmaWithExpertModel()

        time_mlp_model_path = ARTIFACT_DIR / "time_mlp.hmm"
        action_in_proj_model_path = ARTIFACT_DIR / "action_in_proj.hmm"
        action_out_proj_model_path = ARTIFACT_DIR / "action_out_proj.hmm"

        self.action_in_proj_model = tcim.runtime.load(str(action_in_proj_model_path))
        self.action_out_proj_model = tcim.runtime.load(str(action_out_proj_model_path))
        self.time_mlp_model = tcim.runtime.load(str(time_mlp_model_path))
        self.action_in_proj_out_shape = self.action_in_proj_model.get_output_info(
            "action_in_proj_out"
        ).shape
        self.action_in_proj_out_features = self.action_in_proj_out_shape[-1]

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def action_in_proj(self, noisy_actions: Tensor):
        if noisy_actions.dtype != torch.float16:
            noisy_actions = noisy_actions.half()
        t0 = time.time()
        self.action_in_proj_model.set_input(
            "action_in", noisy_actions.contiguous().detach().cpu().numpy()
        )
        t1 = time.time()
        self.action_in_proj_model.run()
        self.action_in_proj_model.sync()
        t2 = time.time()
        out = self.action_in_proj_model.get_output("action_in_proj_out").numpy()
        t3 = time.time()
        if PRINT_DEBUG:
            print(
                f"[Pi05][action_in_proj] setinput: {(t1-t0)*1000:.3f}ms, infer: {(t2-t1)*1000:.3f}ms, getoutput: {(t3-t2)*1000:.3f}ms, total: {(t3-t0)*1000:.3f}ms"
            )
        return torch.from_numpy(out)

    def action_out_proj(self, suffix_out: Tensor):
        if suffix_out.dtype != torch.float16:
            suffix_out = suffix_out.half()
        t0 = time.time()
        self.action_out_proj_model.set_input(
            "action_out", suffix_out.contiguous().detach().cpu().numpy()
        )
        t1 = time.time()
        self.action_out_proj_model.run()
        self.action_out_proj_model.sync()
        t2 = time.time()
        out = self.action_out_proj_model.get_output("action_out_proj_out").numpy()
        t3 = time.time()
        if PRINT_DEBUG:
            print(
                f"[Pi05][action_out_proj] setinput: {(t1-t0)*1000:.3f}ms, infer: {(t2-t1)*1000:.3f}ms, getoutput: {(t3-t2)*1000:.3f}ms, total: {(t3-t0)*1000:.3f}ms"
            )
        return torch.from_numpy(out)

    def time_mlp(self, time_emb: Tensor):
        if time_emb.dtype != torch.float16:
            time_emb = time_emb.half()
        t0 = time.time()
        self.time_mlp_model.set_input(
            "time_emb", time_emb.contiguous().detach().cpu().numpy()
        )
        t1 = time.time()
        self.time_mlp_model.run()
        self.time_mlp_model.sync()
        t2 = time.time()
        out = self.time_mlp_model.get_output("time_mlp_out").numpy()
        t3 = time.time()
        if PRINT_DEBUG:
            print(
                f"[Pi05][time_mlp] setinput: {(t1-t0)*1000:.3f}ms, infer: {(t2-t1)*1000:.3f}ms, getoutput: {(t3-t2)*1000:.3f}ms, total: {(t3-t0)*1000:.3f}ms"
            )
        return torch.from_numpy(out)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = lang_embed_func(tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj_out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = self.time_mlp(time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device
        )

    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.valid_kvcache_length = 0
        # gemma2b prefill
        self.paligemma_with_expert.prefill(
            attention_mask=prefix_att_2d_masks_4d,
            inputs_embeds=prefix_embs,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(
                bsize
            )

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        x_t,
        timestep,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        pad_len = 1024 - full_att_2d_masks_4d.size(-1)
        if pad_len > 0:
            full_att_2d_masks_4d = torch.nn.functional.pad(
                full_att_2d_masks_4d, (0, pad_len), value=torch.finfo(torch.float32).min
            )
        # full_att_2d_masks_4d = full_att_2d_masks_4d.expand(
        #     full_att_2d_masks_4d.size(0),
        #     8,
        #     full_att_2d_masks_4d.size(2),
        #     full_att_2d_masks_4d.size(3),
        # )

        suffix_out = self.paligemma_with_expert.decode(
            attention_mask=full_att_2d_masks_4d,
            inputs_embeds=suffix_embs,
            adarms_cond=adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        return self.action_out_proj(suffix_out)


class HMPI05Policy(PreTrainedPolicy):
    config_class = HMPI05Config
    name = "hm_pi05"

    def __init__(self, config: HMPI05Config):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI05PolicyXH2a(config, self.rtc_processor)

        self.reset()

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ):
        config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=False,
            resume_download=None,
            proxies=None,
            token=None,
            cache_dir=None,
            local_files_only=False,
            revision=None,
            **kwargs,
        )
        model = cls(config, **kwargs)
        return model

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        raise NotImplementedError

    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = (
            batch[f"{OBS_LANGUAGE_TOKENS}"],
            batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"],
        )
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert (
            not self._rtc_enabled()
        ), "RTC is not supported for select_action, use it with predict_action_chunk"

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def _rtc_enabled(self) -> bool:
        return self.model._rtc_enabled()

    def _preprocess_images(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """

        images = []
        img_masks = []

        device = "cpu"

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [
            key for key in self.config.image_features if key not in batch
        ]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = (
                img.shape[1] == 3
            )  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
