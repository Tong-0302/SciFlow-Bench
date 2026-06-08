import os
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import argparse
from pathlib import Path
from diffusers import DiffusionPipeline


# ============================================================
# 默认参数
# ============================================================
DEFAULT_SEED = 42
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# ============================================================
# 相对路径
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"      # ★ 所有 HF 模型都缓存到这里


# ============================================================
# 加载 SDXL
# ============================================================
def load_sdxl_pipeline():
    """
    行为说明：
    1. 第一次运行：
       - 通过 hf-mirror 下载 SDXL
       - 下载内容落在 ./models/ 下（HF cache 结构）
    2. 后续运行：
       - 直接从 ./models/ 复用
       - 无需联网
    3. 整个 models/ 目录可以直接 rsync / scp 到其他机器使用
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading SDXL ===")
    print(f"[Model ID] {MODEL_ID}")
    print(f"[Cache Dir] {MODELS_DIR.resolve()}")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=str(MODELS_DIR),   # ★ 关键：相对路径缓存
    )

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    return pipe


# ============================================================
# 单张生成
# ============================================================
def generate_image(pipe, prompt, out_path, seed=DEFAULT_SEED):
    generator = torch.Generator("cuda").manual_seed(seed)

    print("🎨 Generating image")
    print(f"[Prompt] {prompt}")
    print(f"[Seed]   {seed}")
    print(f"[Out]    {out_path}")

    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator
    ).images[0]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    print(f"✅ Saved: {out_path}")

def load_sdxl():
    """
    给 baseline_final 用的统一接口
    """
    return load_sdxl_pipeline()


def generate(
    pipe,
    prompt,
    out_path,
    seed=DEFAULT_SEED,
):
    """
    给 baseline_final 用的统一生成接口
    """
    return generate_image(
        pipe=pipe,
        prompt=prompt,
        out_path=out_path,
        seed=seed,
    )
# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SDXL image generation (auto-download, relative cache, hf-mirror)"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    pipe = load_sdxl_pipeline()
    generate_image(pipe, args.prompt, args.out, args.seed)


if __name__ == "__main__":
    main()
