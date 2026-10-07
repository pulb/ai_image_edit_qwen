# SPDX-License-Identifier: LicenseRef-Qwen-Research-License-Agreement
# This file integrates with Qwen-Image-2.1, whose weights are distributed
# under Alibaba's Qwen RESEARCH LICENSE AGREEMENT (non-open-source; see
# https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), not GPL.
"""
Loads and runs the Qwen-Image-2.1 diffusers pipeline directly — no
ComfyUI — on a Hugging Face ZeroGPU worker.

Uses module-level state (the loaded pipeline, AOTI status) rather than
instance state on a class: spaces.GPU wraps a plain function at
import time, and a deployment only ever loads one model per worker
process, so instance state would just carry the same value differently.

Adapted from the backend code in app.py of the reference Space,
https://huggingface.co/spaces/hugging-apps/qwen-image-2-1 (declared
license: qwen-research). Modified: Gradio, the NCII guard and prompt
enhancement are removed, and the loading, duration-estimation and
diffusion code is restructured into module-level functions.
"""
import math
import time
from typing import List, Optional

import spaces
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from ai_image_edit_qwen import aoti

MODEL_ID = "Qwen/Qwen-Image-2.1"

_pipe: Optional[QwenImage21Pipeline] = None
_aoti_loaded: List[str] = []
_aoti_blocks_active = False


def load(aoti_repo: Optional[str], aoti_token: Optional[str], use_aoti: bool) -> None:
    """
    One-time setup: load the pipeline onto CUDA and, if requested and
    available, swap in the AOTI-compiled kernels. Call once at startup.
    """
    global _pipe, _aoti_loaded, _aoti_blocks_active

    print(f"[qwen_image21] loading {MODEL_ID} ...", flush=True)
    _pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    _pipe = _pipe.to("cuda")

    # The VAE decoder upcasts to fp32 in its upsamplers, so a large decode
    # allocates far more than the denoising loop that produced it: on the
    # reference Space this died with an allocator assert *after* all steps
    # had completed, and left the ZeroGPU worker wedged so every later
    # request failed until the Space was restarted. Tiling keeps peak VAE
    # memory flat regardless of output size.
    _pipe.vae.enable_tiling(
        tile_sample_min_height=1536,
        tile_sample_min_width=1536,
        tile_sample_stride_height=1152,
        tile_sample_stride_width=1152,
    )

    if use_aoti and aoti_repo:
        try:
            _aoti_loaded, aoti_config = aoti.aoti_load_pipeline(_pipe, aoti_repo, token=aoti_token)
            print(f"[qwen_image21] aoti kernels from {aoti_repo}: {_aoti_loaded} (torch {aoti_config['torch']})", flush=True)
        except Exception as exc:  # noqa: BLE001 — fall back to eager rather than failing startup
            print(f"[qwen_image21] aoti unavailable, running eager: {exc!r}", flush=True)
            _aoti_loaded = []
    _aoti_blocks_active = "QwenImage21DecodeBlock" in _aoti_loaded
    print("[qwen_image21] pipeline ready", flush=True)


# ---- GPU-second budgeting for spaces.GPU's dynamic reservation ----
# Coefficients fit against [generate] log timings on the reference Space's
# own hardware, with the AOTI kernels live, at 40 steps. These are a
# starting point, not a guarantee — recalibrate against your own logs if
# you're serving this on different hardware or the numbers drift.
EAGER_STEP_PENALTY = 1.3
KV_CACHE_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
KV_CACHE_BUDGET_GB = 10.0


def _prefix_tokens(n_images: int, resolution: int) -> int:
    return n_images * (int(resolution) // 16) ** 2


def kv_cache_fits(n_images: int, resolution: int) -> bool:
    return _prefix_tokens(n_images, resolution) * KV_CACHE_BYTES_PER_TOKEN <= KV_CACHE_BUDGET_GB * 1e9


def _estimate_duration(
    prompt: str = "",
    image_paths: Optional[List[str]] = None,
    negative_prompt: str = "",
    true_cfg_scale: float = 1.0,
    num_inference_steps: int = 40,
    seed: int = 0,
    resolution: int = 1024,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> int:
    """
    spaces.GPU(duration=...) calls this with the exact same arguments
    _diffuse is about to be called with, to decide how many GPU seconds to
    reserve for that call. Its parameter order must therefore match
    _diffuse's below.
    """
    try:
        steps = int(num_inference_steps)
        res = int(resolution)
        cfg = float(true_cfg_scale)
    except (TypeError, ValueError):
        steps, res, cfg = 40, 1024, 1.0
    n_images = len(image_paths or [])
    tokens = (res / 16.0) ** 2
    prefix = _prefix_tokens(n_images, res)
    cached = kv_cache_fits(n_images, res)
    tokens += 0.2 * prefix if cached else prefix
    per_step = 3.4e-6 * tokens ** 1.363
    if cfg > 1.0 and (negative_prompt or "").strip():
        per_step *= 2.0
    if not (_aoti_blocks_active and cached):
        per_step *= EAGER_STEP_PENALTY
    fixed = 4.0 + 2.0 * n_images + 1.5e-6 * res * res
    return int(min(340, math.ceil((fixed + steps * per_step) * 1.25)))


@spaces.GPU(duration=_estimate_duration)
def _diffuse(
    prompt: str,
    image_paths: List[str],
    negative_prompt: str,
    true_cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    resolution: int,
    width: Optional[int],
    height: Optional[int],
) -> Image.Image:
    """Runs the diffusion pipeline. Inputs are already validated by the caller."""
    if _pipe is None:
        raise RuntimeError("qwen_image21 pipeline not loaded — call pipeline.load() first.")

    condition_images = [Image.open(p) for p in image_paths] or None
    negative_prompt = (negative_prompt or "").strip()
    use_kv_cache = kv_cache_fits(len(image_paths), int(resolution))
    call_kwargs = {"use_kv_cache": use_kv_cache}
    if negative_prompt and float(true_cfg_scale) > 1.0:
        call_kwargs["negative_prompt"] = negative_prompt
        call_kwargs["true_cfg_scale"] = float(true_cfg_scale)

    started = time.perf_counter()
    image = _pipe(
        prompt=prompt,
        image=condition_images,
        height=height,
        width=width,
        output_resolution=int(resolution),
        num_inference_steps=int(num_inference_steps),
        generator=torch.Generator(device="cuda").manual_seed(int(seed)),
        **call_kwargs,
    ).images[0]
    print(
        f"[qwen_image21] images={len(image_paths)} steps={num_inference_steps} "
        f"res={resolution} size={image.size} kv_cache={use_kv_cache} "
        f"aoti={_aoti_loaded} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return image


def generate(
    prompt: str,
    image_paths: List[str],
    negative_prompt: str,
    true_cfg_scale: float,
    num_inference_steps: int,
    seed: int,
    resolution: int,
    width: Optional[int],
    height: Optional[int],
) -> Image.Image:
    """Public entry point: runs one generation and returns the image."""
    return _diffuse(
        prompt, image_paths, negative_prompt, true_cfg_scale,
        num_inference_steps, seed, resolution, width, height,
    )
