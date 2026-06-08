import os
import base64
import requests
from pathlib import Path
from openai import OpenAI

# ============================================================
# 配置 (与你提供的脚本一致)
# ============================================================
API_KEY = os.environ.get("DF_API_KEY", "")
BASE_HOST = 'http://localhost:3888/v1'
MODEL_NAME = "doubao-seedream-4-0-250828"

def load_seedream():
    """
    初始化 OpenAI 客户端，返回 client 和 占位符 device
    """
    print(f"=== Initializing Doubao-Seedream API ===")
    client = OpenAI(
        api_key=os.environ.get("SEEDREAM_API_KEY", API_KEY),
        base_url=BASE_HOST
    )
    return client, "api"

def generate(pipe, device, prompt, out_path, seed=42):
    """
    单次生成函数：实现方法与你提供的 process_item 核心逻辑完全一致
    """
    client = pipe
    out_path = Path(out_path)
    
    try:
        # 1. 调用 SDK
        response = client.images.generate(
            model=MODEL_NAME,
            prompt=prompt,
            n=1,
            extra_body={"seed": seed} # 如果 API 支持传 seed
        )

        if not response.data or len(response.data) == 0:
            print(f"❌ [Seedream] No data in response")
            return

        image_obj = response.data[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. 策略 A: 检查 b64_json (保留你脚本的实现)
        if getattr(image_obj, 'b64_json', None) and image_obj.b64_json.strip():
            image_bytes = base64.b64decode(image_obj.b64_json)
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            print(f"✅ [Seedream] Saved via Base64: {out_path.name}")
        
        # 3. 策略 B: 检查 URL (保留你脚本的实现)
        elif getattr(image_obj, 'url', None) and image_obj.url.strip():
            img_resp = requests.get(image_obj.url, timeout=120)
            if img_resp.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(img_resp.content)
                print(f"✅ [Seedream] Saved via URL: {out_path.name}")
            else:
                print(f"❌ [Seedream] URL Download Failed: HTTP {img_resp.status_code}")

    except Exception as e:
        print(f"❌ [Seedream Exception] {str(e)}")