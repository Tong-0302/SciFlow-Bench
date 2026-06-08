import os
import base64
import requests
from pathlib import Path
from openai import OpenAI

# ============================================================
# Qwen 配置 (通常也走 OpenAI 兼容格式)
# ============================================================
API_KEY = os.environ.get("DF_API_KEY", "") 
BASE_HOST = 'http://localhost:3888/v1' # 或者是 DashScope 的地址
MODEL_NAME = "qwen-image-plus" # 或 qwen-plus

def load_qwen():
    print(f"=== Initializing Qwen Image API ===")
    client = OpenAI(
        api_key=os.environ.get("QWEN_API_KEY", API_KEY),
        base_url=BASE_HOST
    )
    return client, "api"

def generate(pipe, device, prompt, out_path, seed=42, **kwargs):
    """
    逻辑与 Seedream 完全一致，确保 baseline_final 调用方式统一
    """
    client = pipe
    try:
        response = client.images.generate(
            model=MODEL_NAME,
            prompt=prompt,
            n=1,
            extra_body={"seed": seed}
        )
        
        if not response.data: return
        image_obj = response.data[0]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Base64 策略
        if getattr(image_obj, 'b64_json', None) and image_obj.b64_json.strip():
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(image_obj.b64_json))
            print(f"✅ [Qwen] Saved: {out_path.name}")
        # URL 策略
        elif getattr(image_obj, 'url', None) and image_obj.url.strip():
            img_resp = requests.get(image_obj.url, timeout=120)
            if img_resp.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(img_resp.content)
                print(f"✅ [Qwen] Saved: {out_path.name}")
    except Exception as e:
        print(f"❌ [Qwen Exception] {str(e)}")