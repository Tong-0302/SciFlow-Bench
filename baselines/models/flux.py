import os
import torch
import argparse
from pathlib import Path

# ============================================================
# ⚠️ 关键补丁：兼容旧版 PyTorch（吞掉 enable_gqa）
# 必须在 import diffusers 之前
# ============================================================
# import torch.nn.functional as F

# _orig_sdp = F.scaled_dot_product_attention

# def _sdp_compat(
#     query,
#     key,
#     value,
#     attn_mask=None,
#     dropout_p=0.0,
#     is_causal=False,
#     scale=None,
#     enable_gqa=None,   # ← 吃掉这个参数
#     **kwargs
# ):
#     return _orig_sdp(
#         query,
#         key,
#         value,
#         attn_mask=attn_mask,
#         dropout_p=dropout_p,
#         is_causal=is_causal,
#         scale=scale,
#     )

# F.scaled_dot_product_attention = _sdp_compat
# ============================================================

from diffusers import FluxPipeline


# ============================================================
# 默认参数
# ============================================================
DEFAULT_SEED = 42
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 3.5
DEFAULT_RESOLUTION = 1024


# ============================================================
# 基于脚本位置的【相对路径】模型目录
# AAA_Experiment/models/FLUX.1-dev
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models" / "FLUX.1-dev"


# ============================================================
# 内部加载实现（原样保留）
# ============================================================
def load_flux_pipeline():
    print("=== Loading FLUX.1-dev ===")
    print(f"[Model Path] {MODEL_DIR}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"""
❌ FLUX model not found.

Expected directory:
  {MODEL_DIR}

Please make sure the model exists at:
  AAA_Experiment/models/FLUX.1-dev
"""
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        local_files_only=True,   # ★ 不允许联网
    )

    #pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)

    print(f"[Ready] FLUX loaded on {device}")
    return pipe


# ============================================================
# baseline_final 用的统一加载接口（alias）
# ============================================================
def load_flux():
    """
    给 baseline_final / pipeline 用的统一接口
    """
    return load_flux_pipeline()


# ============================================================
# 内部生成实现（原样保留）
# ============================================================
def generate_image(
    pipe,
    prompt,
    out_path,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    guidance=DEFAULT_GUIDANCE,
):
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed)

    print("🎨 Generating image")
    print(f"[Prompt]     {prompt}")
    print(f"[Seed]       {seed}")
    print(f"[Steps]      {steps}")
    print(f"[Guidance]   {guidance}")
    print(f"[Resolution] {DEFAULT_RESOLUTION} x {DEFAULT_RESOLUTION}")
    print(f"[Out]        {out_path}")

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=DEFAULT_RESOLUTION,
            width=DEFAULT_RESOLUTION,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
        ).images[0]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    print(f"✅ Saved: {out_path}")


# ============================================================
# baseline_final 用的统一生成接口（alias）
# ============================================================
def generate(
    pipe,
    prompt,
    out_path,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    guidance=DEFAULT_GUIDANCE,
):
    """
    给 baseline_final 用：
    - 使用已加载 pipe
    - 不负责 reload（Flux 一般不需要 fp32 fallback）
    """
    return generate_image(
        pipe=pipe,
        prompt=prompt,
        out_path=out_path,
        seed=seed,
        steps=steps,
        guidance=guidance,
    )


# ============================================================
# CLI（保持可单独运行）
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1-dev image generation (local model, relative path, torch-compatible)"
    )
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    args = parser.parse_args()

    pipe = load_flux_pipeline()
    generate_image(
        pipe=pipe,
        prompt=args.prompt,
        out_path=args.out,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
