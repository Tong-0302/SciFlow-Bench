import os
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import argparse
from pathlib import Path
from diffusers import PixArtSigmaPipeline

# ============================================================
# 默认参数
# ============================================================
DEFAULT_SEED = 42
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 4.5
DEFAULT_RESOLUTION = 1024

MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

# ============================================================
# 相对路径模型缓存（★关键）
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"      # 整个目录可拷走


# ============================================================
# 加载 PixArt-Sigma（首次自动下载，之后复用）
# ============================================================
def load_pixart_pipeline():
    """
    行为说明：
    1. 第一次运行：
       - 通过 hf-mirror 下载 PixArt-Sigma
       - 缓存到 ./models/（HF 标准 cache 结构）
    2. 后续运行：
       - 直接从 ./models/ 复用
       - 不再联网
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print("=== Loading PixArt-Sigma ===")
    print(f"[Model ID] {MODEL_ID}")
    print(f"[Cache Dir] {MODELS_DIR.resolve()}")
    print(f"[Device] {device}, dtype={torch_dtype}")

    pipe = PixArtSigmaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        cache_dir=str(MODELS_DIR),   # ★ 相对路径缓存
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe, device


# ============================================================
# 单张生成（无 negative prompt）
# ============================================================
def generate(
    pipe,
    device,
    prompt,
    out_path,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    guidance_scale=DEFAULT_GUIDANCE
):
    print("🎨 Generating image")
    print(f"[Prompt] {prompt}")
    print(f"[Seed] {seed}")
    print(f"[Steps] {steps}")
    print(f"[Guidance] {guidance_scale}")
    print(f"[Resolution] {DEFAULT_RESOLUTION} x {DEFAULT_RESOLUTION}")
    print(f"[Out] {out_path}")

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=DEFAULT_RESOLUTION,
            width=DEFAULT_RESOLUTION,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    print(f"✅ Saved: {out_path}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="PixArt-Sigma generation (auto-download, relative cache, hf-mirror)"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    args = parser.parse_args()

    pipe, device = load_pixart_pipeline()
    generate(
        pipe=pipe,
        device=device,
        prompt=args.prompt,
        out_path=args.out,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance
    )


if __name__ == "__main__":
    main()

# import os

# # ============================================================
# # 必须最早设置：尽量阻止 transformers/apex 走 fused 路径
# # ============================================================
# os.environ.setdefault("TRANSFORMERS_NO_APEX", "1")
# os.environ.setdefault("APEX_DISABLE_FUSED_LAYER_NORM", "1")

# # HuggingFace 镜像
# os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# import torch
# import argparse
# from pathlib import Path
# from diffusers import PixArtSigmaPipeline

# # ============================================================
# # 默认参数
# # ============================================================
# DEFAULT_SEED = 42
# DEFAULT_STEPS = 20
# DEFAULT_GUIDANCE = 4.5
# DEFAULT_RESOLUTION = 1024

# MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

# # ============================================================
# # 相对路径模型缓存（整个 models 目录可拷走）
# # ============================================================
# BASE_DIR = Path(__file__).parent.resolve()
# MODELS_DIR = BASE_DIR / "models"


# # ============================================================
# # Utils
# # ============================================================
# def _pick_device():
#     return "cuda" if torch.cuda.is_available() else "cpu"


# def _pick_dtype(device: str):
#     # GPU → fp16，CPU → fp32
#     return torch.float16 if device == "cuda" else torch.float32


# def _should_fallback_to_fp32(err: Exception) -> bool:
#     msg = str(err)
#     return (
#         ("expected scalar type Float but found Half" in msg)
#         or ("fused_rms_norm" in msg)
#         or ("fused_layer_norm" in msg)
#         or ("FusedRMSNorm" in msg)
#     )


# # ============================================================
# # 加载 PixArt-Sigma（统一入口，baseline_final 用）
# # ============================================================
# def load_pixart():
#     """
#     统一加载接口：
#     - baseline_final / pipeline 只调用这个
#     - dtype / device 策略完全内聚
#     """
#     MODELS_DIR.mkdir(parents=True, exist_ok=True)

#     device = _pick_device()
#     dtype = _pick_dtype(device)

#     print("=== Loading PixArt-Sigma ===")
#     print(f"[Model ID]  {MODEL_ID}")
#     print(f"[Cache Dir] {MODELS_DIR.resolve()}")
#     print(f"[Device]    {device}, dtype={dtype}")

#     pipe = PixArtSigmaPipeline.from_pretrained(
#         MODEL_ID,
#         torch_dtype=dtype,
#         use_safetensors=True,
#         cache_dir=str(MODELS_DIR),
#     )

#     pipe = pipe.to(device)
#     pipe.set_progress_bar_config(disable=False)
#     return pipe, device


# # ============================================================
# # 单次生成（不负责 fallback）
# # ============================================================
# def _generate_once(pipe, device, prompt, out_path, seed, steps, guidance_scale):
#     print("🎨 Generating image")
#     print(f"[Prompt]     {prompt}")
#     print(f"[Seed]       {seed}")
#     print(f"[Steps]      {steps}")
#     print(f"[Guidance]   {guidance_scale}")
#     print(f"[Resolution] {DEFAULT_RESOLUTION} x {DEFAULT_RESOLUTION}")
#     print(f"[Out]        {out_path}")

#     generator = torch.Generator(device=device).manual_seed(seed)

#     with torch.inference_mode():
#         image = pipe(
#             prompt=prompt,
#             height=DEFAULT_RESOLUTION,
#             width=DEFAULT_RESOLUTION,
#             num_inference_steps=steps,
#             guidance_scale=guidance_scale,
#             generator=generator,
#         ).images[0]

#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     image.save(out_path)
#     print(f"✅ Saved: {out_path}")


# # ============================================================
# # baseline_final 用的生成接口（不 reload）
# # ============================================================
# def generate(
#     pipe,
#     device,
#     prompt,
#     out_path,
#     seed=DEFAULT_SEED,
#     steps=DEFAULT_STEPS,
#     guidance=DEFAULT_GUIDANCE,
# ):
#     """
#     给 baseline_final 用：
#     - 使用已加载 pipe
#     - 如果 fp16 生成阶段炸了 → 自动降级 fp32
#     """
#     try:
#         _generate_once(pipe, device, prompt, out_path, seed, steps, guidance)
#     except RuntimeError as e:
#         if not _should_fallback_to_fp32(e):
#             raise

#         print("\n[Warn] Detected Apex/LayerNorm Float/Half mismatch.")
#         print("       Reload PixArt with fp32 and retry once...\n")

#         # 清理
#         del pipe
#         if device == "cuda":
#             torch.cuda.empty_cache()

#         # fp32 重载
#         pipe, device = _reload_fp32()
#         _generate_once(pipe, device, prompt, out_path, seed, steps, guidance)


# def _reload_fp32():
#     device = _pick_device()

#     print("=== Reloading PixArt-Sigma (fp32) ===")
#     pipe = PixArtSigmaPipeline.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.float32,
#         use_safetensors=True,
#         cache_dir=str(MODELS_DIR),
#     )

#     pipe = pipe.to(device)
#     pipe.set_progress_bar_config(disable=False)
#     return pipe, device


# # ============================================================
# # CLI（保持你原来的行为）
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser(
#         description="PixArt-Sigma generation (auto-download, relative cache, auto fp32 fallback)"
#     )
#     parser.add_argument("--prompt", required=True)
#     parser.add_argument("--out", required=True)
#     parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
#     parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
#     parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
#     args = parser.parse_args()

#     pipe, device = load_pixart()
#     generate(
#         pipe=pipe,
#         device=device,
#         prompt=args.prompt,
#         out_path=args.out,
#         seed=args.seed,
#         steps=args.steps,
#         guidance=args.guidance,
#     )


# if __name__ == "__main__":
#     main()

