"""
run_pdf_infer_ocr.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
功能:
  调用 DeepSeek-OCR 模型，将 PDF 转换为图片并执行 OCR，
  输出 Markdown 文件与分割后的图像，支持从环境变量指定路径。

环境变量:
  PDF_FILE   -> 输入 PDF 文件路径
  OUTPUT_DIR -> 输出目录路径（将自动转换为绝对路径）

工作流调用:
  subprocess.run(["python", "dataflow_agent/toolkits/run_pdf_infer_ocr.py"], env=env)
"""

import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# === 强制设置自定义 HF 缓存目录 (必须在任何 HF import 之前) ===
# CUSTOM_CACHE = BASE_DIR / "huggingface" / "download"

# os.environ["HF_HOME"] = str(CUSTOM_CACHE)
# os.environ["TRANSFORMERS_CACHE"] = str(CUSTOM_CACHE)
# os.environ["HUGGINGFACE_HUB_CACHE"] = str(CUSTOM_CACHE)

# 避免残留设置污染
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

# print(f"[HF Cache] Using custom cache directory: {CUSTOM_CACHE}")
import re
import torch
import shutil
import tempfile
import requests
import contextlib
from pathlib import Path
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 1. 路径与环境配置
# ============================================================
pdf_file = os.getenv("PDF_FILE", "/path/to/your/paper.pdf")
raw_output_dir = os.getenv("OUTPUT_DIR", "./output/merged_dir")
output_dir = os.path.abspath(raw_output_dir)

os.makedirs(output_dir, exist_ok=True)
print(f"📂 输出目录: {output_dir}")

# ============================================================
# 2. Hugging Face 镜像配置
# ============================================================
def _set_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def _unset_offline():
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(os.environ["HF_HOME"])
_unset_offline()
print("🌐 Checking Hugging Face mirror connectivity...")
mirror_ok = False
with contextlib.suppress(Exception):
    r = requests.get("https://hf-mirror.com", timeout=4)
    if r.status_code == 200:
        mirror_ok = True

if mirror_ok:
    print("✅ Mirror available: using hf-mirror.com for downloads.")
else:
    print("⚠️ Mirror unreachable — switching to offline mode.")
    _set_offline()

import huggingface_hub
from huggingface_hub import file_download

huggingface_hub.constants.HF_ENDPOINT = "https://hf-mirror.com"
old_hf = "https://huggingface.co"
mirror_hf = "https://hf-mirror.com"

if hasattr(file_download, "_CACHED_HF_ENDPOINT"):
    file_download._CACHED_HF_ENDPOINT = mirror_hf
file_download.HUGGINGFACE_CO_URL_TEMPLATE = file_download.HUGGINGFACE_CO_URL_TEMPLATE.replace(old_hf, mirror_hf)
file_download.HUGGINGFACE_CO_RESOLVE_ENDPOINT = mirror_hf
print(f"🔧 HuggingFace endpoint fully patched to hf-mirror.com")

# ============================================================
# 3. 模型加载
# ============================================================

model_name = 'deepseek-ai/DeepSeek-OCR'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading model: {model_name} on device: {device}")

if device == "cpu":
    print("⚠️ Warning: Running DeepSeek-OCR on CPU will be extremely slow!")
print(bool(os.getenv("HF_HUB_OFFLINE", "0") == "1"))
print(os.getenv("HF_HOME"))
print(f"🚀 Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True,
    # use_fast=False,
    local_files_only=bool(os.getenv("HF_HUB_OFFLINE", "0") == "1")
)
print("tokenizer loaded")
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True,
    local_files_only=bool(os.getenv("HF_HUB_OFFLINE", "0") == "1")
)
print("model loaded")

model = model.eval().to(device).to(torch.bfloat16)
print("✅ Model ready on GPU (bfloat16 mode).")

# ============================================================
# 4. 参数配置
# ============================================================
prompt = "<image>\n<|grounding|>Convert the document to markdown."
base_size = 1024
image_size = 640
crop_mode = True
test_compress = True


# ============================================================
# 5. PDF → 图片
# ============================================================
def pdf_to_images(pdf_path, dpi=300):
    temp_dir = tempfile.TemporaryDirectory()
    temp_image_dir = Path(temp_dir.name)
    images = convert_from_path(
        pdf_path, dpi=dpi,
        output_folder=temp_image_dir,
        fmt='jpg', thread_count=4,
        use_pdftocairo=True
    )
    image_paths = sorted([str(p) for p in temp_image_dir.glob("*.jpg")])[:8]
    print(f"📘 PDF转换完成，共生成 {len(image_paths)} 张图片")
    return image_paths, temp_dir


def extract_page_number(path: str) -> int:
    stem = Path(path).stem
    m = re.search(r'-(\d+)$', stem)
    return int(m.group(1)) if m else 0


# ============================================================
# 6. 主处理逻辑
# ============================================================
def process_pdf_ocr(pdf_file: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_paths, temp_dir = pdf_to_images(pdf_file)
    try:
        image_paths = sorted(image_paths, key=extract_page_number)
        all_markdown = []

        for page_idx, img_path in enumerate(image_paths, start=1):
            print(f"\n🧩 正在处理第 {page_idx}/{len(image_paths)} 页 ({os.path.basename(img_path)}) ...")
            temp_page_dir = tempfile.mkdtemp(dir=output_dir, prefix=f"temp_page_{page_idx}_")

            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=img_path,
                output_path=temp_page_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=test_compress
            )

            page_prefix = f"page{page_idx}"

            # ---- 处理子图 ----
            image_dir = os.path.join(temp_page_dir, "images")
            if os.path.exists(image_dir):
                for img_file in sorted(Path(image_dir).glob("*.jpg")):
                    img_idx = re.search(r'(\d+)', img_file.stem)
                    img_idx = img_idx.group(1) if img_idx else "0"
                    dst = os.path.join(output_dir, f"{page_prefix}_{img_idx}.jpg")
                    shutil.move(str(img_file), dst)

            # ---- 处理 result.mmd ----
            mmd_src = os.path.join(temp_page_dir, "result.mmd")
            mmd_dst = os.path.join(output_dir, f"{page_prefix}_result.mmd")
            if os.path.exists(mmd_src):
                shutil.move(mmd_src, mmd_dst)
                with open(mmd_dst, 'r', encoding='utf-8') as f:
                    page_markdown = f.read()

                page_markdown = re.sub(
                    r'!\s*\[\s*[^\]]*\s*\]\s*\(\s*(?:\.?/)?images/(\d+\.(?:jpg|png|jpeg))\s*\)',
                    fr'![](images/page{page_idx}_\1)', page_markdown
                )
                all_markdown.append(f"## 第 {page_idx} 页\n\n{page_markdown}")
            else:
                all_markdown.append(f"### 第 {page_idx} 页：未找到 result.mmd 文件")

            shutil.rmtree(temp_page_dir)

        # ---- 合并 markdown ----
        final_markdown = "\n\n---\n\n".join(all_markdown)
        final_output_path = os.path.join(output_dir, "merged_result.md")
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)

        print(f"\n✅ OCR处理完成！结果已按页码顺序保存到：{output_dir}")

    finally:
        temp_dir.cleanup()
        print("🧹 临时图片文件已清理完毕。")


# ============================================================
# 7. CLI 调用入口
# ============================================================
if __name__ == "__main__":
    try:
        process_pdf_ocr(pdf_file, output_dir)
    except Exception as e:
        print(f"❌ OCR 处理失败: {e}")
        raise
