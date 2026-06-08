import os
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default=None, help="要做 OCR 的图片路径")
parser.add_argument("--out_dir", type=str, default="./output/dir", help="DeepSeek-OCR 自己的输出目录")
args = parser.parse_args()
image_file = args.img
output_path = args.out_dir

# 与 PDF OCR 工作流保持一致
HF_CACHE = BASE_DIR / "huggingface" / "download"

# 设置所有可能的 HF 缓存环境变量
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] =str(HF_CACHE)

# 关键补丁：避免 root 用户 HOME 覆盖缓存路径
os.environ["HOME"] = str(HF_CACHE)

# 避免 offline 模式
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

print(f"📂 使用 HF Cache 目录: {HF_CACHE}")

# ============================================================
# 1. 在设置环境变量之后，才 import transformers
# ============================================================
from transformers.utils import hub
hub.HF_HOME = str(HF_CACHE)
hub.TRANSFORMERS_CACHE = str(HF_CACHE)
hub.HUGGINGFACE_HUB_CACHE = str(HF_CACHE)

from transformers import AutoModel, AutoTokenizer
import torch, requests, contextlib, huggingface_hub
from huggingface_hub import file_download, snapshot_download


# ============================================================
# 2. 你的原来代码逻辑（镜像、模型加载等）保持不变
# ============================================================

# ---- 镜像选择 ----
def patch_hf_endpoint(endpoint: str):
    huggingface_hub.constants.HF_ENDPOINT = endpoint
    if hasattr(file_download, "_CACHED_HF_ENDPOINT"):
        file_download._CACHED_HF_ENDPOINT = endpoint

    file_download.HUGGINGFACE_CO_URL_TEMPLATE = (
        file_download.HUGGINGFACE_CO_URL_TEMPLATE.replace(
            "https://huggingface.co", endpoint
        )
    )
    file_download.HUGGINGFACE_CO_RESOLVE_ENDPOINT = endpoint
    print(f"🔧 已切换 HuggingFace endpoint 到: {endpoint}")

def try_connect(url, timeout=5):
    with contextlib.suppress(Exception):
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    return False

mirror_list = ["https://hf-mirror.com", "https://modelscope.cn", "https://huggingface.co"]
endpoint_used = None

for m in mirror_list:
    print(f"🌐 尝试连接 {m} ...")
    if try_connect(m):
        endpoint_used = m
        break

if endpoint_used:
    os.environ["HF_ENDPOINT"] = endpoint_used
    patch_hf_endpoint(endpoint_used)
else:
    print("⚠️ 无可用镜像，进入离线模式")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

# ============================================================
# 3. 模型加载
# ============================================================
model_name = "deepseek-ai/DeepSeek-OCR"
local_only = False  # 不强制离线，让模型自动使用缓存

print(f"🚀 加载模型: {model_name}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=local_only
)

# 加载模型
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
   _attn_implementation="flash_attention_2",
    use_safetensors=True,
    local_files_only=local_only
)

model = model.eval().cuda().to(torch.bfloat16)
try:
    snapshot_path = snapshot_download(
        model_name,
        local_files_only=True,  # 只查本地，不会重新下载
    )
    print("📂 本地缓存快照目录:", snapshot_path)
except Exception as e:
    print("⚠️ 无法定位本地缓存快照:", repr(e))

print("✅ DeepSeek-OCR 模型加载完毕（使用统一缓存目录）")

# ============================================================
# 4. 推理配置
# ============================================================
prompt = "<image>\n<|grounding|>OCR this image."

base_size = 1024
image_size = 640
crop_mode = False
test_compress = True
save_results = True

# ============================================================
# 5. 推理执行
# ============================================================
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=base_size,
    image_size=image_size,
    crop_mode=crop_mode,
    save_results=save_results,
    test_compress=test_compress
)

print(f"\n✅ OCR完成！结果已保存到: {output_path}")