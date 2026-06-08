import os
import torch
import argparse
from pathlib import Path
from diffusers import FluxPipeline

# ============================================================
# 默认参数
# ============================================================
DEFAULT_SEED = 42


# ============================================================
# 相对路径定义（★统一风格）
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
# FLUX_DIR = MODELS_DIR / "FLUX.2-dev"
FLUX_DIR = "diffusers/FLUX.2-dev-bnb-4bit"


# ============================================================
# 加载 FLUX.2（只用本地，不联网）
# ============================================================
def load_flux2():
    print("=== Loading FLUX.2-dev ===")
    print(f"[Model Path] {FLUX_DIR}")

#     if not FLUX_DIR.exists():
#         raise FileNotFoundError(
#             f"""
# ❌ FLUX.2 model not found.

# Expected path:
#   {FLUX_DIR}

# Please place the model here:
#   AAA_Experiment/models/FLUX.2-dev

# (This loader is local-only and will NOT download.)
# """
#         )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = FluxPipeline.from_pretrained(
        str(FLUX_DIR),
        torch_dtype=torch_dtype,
        local_files_only=True   # ★ 明确：不联网
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"[Ready] FLUX.2 loaded on {device}")
    return pipe


# ============================================================
# 单张生成
# ============================================================
def generate_image(pipe, prompt, out_path, seed=DEFAULT_SEED):
    print("🎨 Generating image")
    print(f"[Prompt] {prompt}")
    print(f"[Seed]   {seed}")
    print(f"[Out]    {out_path}")

    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,      # FLUX 推荐区间
            num_inference_steps=20,  # ⚠️ FLUX.2 建议较低
            generator=generator,
            max_sequence_length=512
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
        description="FLUX.2-dev image generation (local-only, relative path)"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for Flux generation"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output image path, e.g. outputs/flux2.png"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    args = parser.parse_args()

    pipe = load_flux2()
    generate_image(pipe, args.prompt, args.out, args.seed)


if __name__ == "__main__":
    main()
