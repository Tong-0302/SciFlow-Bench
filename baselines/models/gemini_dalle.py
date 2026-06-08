import os
import base64
import re
import argparse
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------
# Logger
# -------------------------------------------------------------
def log_info(*a): print("[INFO]", *a)
def log_error(*a): print("[ERROR]", *a)

# -------------------------------------------------------------
# Base64 提取
# -------------------------------------------------------------
_B64_RE = re.compile(r"[A-Za-z0-9+/=]+")

def extract_base64(s: str) -> str:
    s = "".join(s.split())
    matches = _B64_RE.findall(s)
    return max(matches, key=len) if matches else ""

# -------------------------------------------------------------
# 显式 timeout 配置（工程级）
# -------------------------------------------------------------
HTTPX_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=300.0,   # 👈 关键：Gemini-3 生图必须拉大
    write=10.0,
    pool=10.0,
)

# -------------------------------------------------------------
# HTTP POST /chat/completions
# -------------------------------------------------------------
async def _post_chat_completions(api_url, api_key, payload):
    url = f"{api_url}/chat/completions".rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    log_info(f"POST {url}")

    async with httpx.AsyncClient(
        timeout=HTTPX_TIMEOUT,
        http2=False
    ) as client:
        resp = await client.post(url, headers=headers, json=payload)
        log_info(f"status={resp.status_code}")
        resp.raise_for_status()
        return resp.json()

# -------------------------------------------------------------
# 模型类型判断
# -------------------------------------------------------------
def _is_dalle_model(model: str) -> bool:
    return model.lower().startswith("dall-e")

def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()

# -------------------------------------------------------------
# DALL·E 生图（无需 retry）
# -------------------------------------------------------------
async def call_dalle_image_generation_async(
    api_url, api_key, model, prompt,
    size="1024x1024",
    quality="standard",
    style="vivid",
    response_format="b64_json"
):
    url = f"{api_url}/images/generations".rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": response_format,
    }

    if model == "dall-e-3":
        payload["quality"] = quality
        payload["style"] = style

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["b64_json"]

# -------------------------------------------------------------
# Gemini 生图（基础版）
# -------------------------------------------------------------
async def call_gemini_image_generation_async(
    api_url, api_key, model, prompt
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "image"},
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    data = await _post_chat_completions(api_url, api_key, payload)
    return data["choices"][0]["message"]["content"]

# -------------------------------------------------------------
# Gemini 生图（带 retry 的工程级封装）
# -------------------------------------------------------------
async def call_gemini_with_retry(
    api_url,
    api_key,
    model,
    prompt,
    retries=3,
    backoff_sec=2.0
):
    for attempt in range(1, retries + 1):
        try:
            return await call_gemini_image_generation_async(
                api_url, api_key, model, prompt
            )
        except httpx.ReadTimeout:
            log_error(f"Gemini ReadTimeout (attempt {attempt}/{retries})")
        except httpx.HTTPError as e:
            log_error(f"Gemini HTTP error (attempt {attempt}/{retries}): {e}")

        if attempt < retries:
            await asyncio.sleep(backoff_sec * attempt)

    raise RuntimeError("Gemini image generation failed after retries")

# -------------------------------------------------------------
# 统一生成函数
# -------------------------------------------------------------
async def generate_and_save_image(
    prompt: str,
    out_path: str,
    model: str,
    api_url: str,
    api_key: str
):
    log_info(f"Model = {model}")
    log_info(f"Prompt = {prompt}")

    if _is_dalle_model(model):
        b64 = await call_dalle_image_generation_async(
            api_url, api_key, model, prompt
        )

    elif _is_gemini_model(model):
        raw = await call_gemini_with_retry(
            api_url, api_key, model, prompt
        )
        b64 = extract_base64(raw)

    else:
        raise ValueError(f"Unsupported model: {model}")

    if not b64:
        raise RuntimeError("无法提取 Base64 图像数据")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))

    log_info(f"Saved → {out_path}")

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate image using Gemini or DALL·E (engineered timeout & retry)"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemini2.5", "gemini3", "dalle"]
    )

    args = parser.parse_args()

    API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
    API_KEY = os.getenv("DF_API_KEY")

    MODEL_NAME_MAP = {
        "gemini2.5": "gemini-2.5-flash-image-preview",
        "dalle": "dall-e-3",
    }

    model_name = MODEL_NAME_MAP[args.model]

    asyncio.run(
        generate_and_save_image(
            prompt=args.prompt,
            out_path=args.out,
            model=model_name,
            api_url=API_URL,
            api_key=API_KEY,
        )
    )

if __name__ == "__main__":
    main()


# import os
# import base64
# import argparse
# import asyncio
# import httpx
# from dotenv import load_dotenv

# load_dotenv()

# # -------------------------------------------------------------
# # Logger
# # -------------------------------------------------------------
# def log_info(*a): print("[INFO]", *a)
# def log_error(*a): print("[ERROR]", *a)

# # -------------------------------------------------------------
# # Timeout（Gemini image 必须大）
# # -------------------------------------------------------------
# HTTPX_TIMEOUT = httpx.Timeout(
#     connect=10.0,
#     read=600.0,
#     write=10.0,
#     pool=10.0,
# )

# # -------------------------------------------------------------
# # Unified image generation (Gemini / DALL·E)
# # -------------------------------------------------------------
# async def call_image_generation_async(
#     api_url: str,
#     api_key: str,
#     model: str,
#     prompt: str,
# ):
#     url = f"{api_url}/images/generations"
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "model": model,
#         "prompt": prompt,
#     }

#     log_info(f"POST {url}  model={model}")

#     async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
#         resp = await client.post(url, headers=headers, json=payload)
#         log_info(f"status={resp.status_code}")
#         resp.raise_for_status()
#         data = resp.json()
#         if isinstance(data, dict) and "data" in data:
#             return data["data"][0]["b64_json"]

#         log_error("Unexpected image generation response:")
#         log_error(data)

#         raise RuntimeError("Image generation failed: unexpected response format")

# # -------------------------------------------------------------
# # Generate & save
# # -------------------------------------------------------------
# async def generate_and_save_image(
#     prompt: str,
#     out_path: str,
#     model: str,
#     api_url: str,
#     api_key: str,
# ):
#     log_info(f"Model  = {model}")
#     log_info(f"Prompt = {prompt}")

#     b64 = await call_image_generation_async(
#         api_url, api_key, model, prompt
#     )

#     os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
#     with open(out_path, "wb") as f:
#         f.write(base64.b64decode(b64))

#     log_info(f"Saved → {out_path}")

# # -------------------------------------------------------------
# # CLI
# # -------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate image using Gemini / DALL·E (APIYI proxy)"
#     )
#     parser.add_argument("--prompt", required=True)
#     parser.add_argument("--out", required=True)
#     parser.add_argument(
#         "--model",
#         required=True,
#         choices=["gemini2.5", "gemini3", "gpt"]
#     )

#     args = parser.parse_args()

#     API_URL = os.getenv("DF_API_URL", "https://api.apiyi.com/v1")
#     API_KEY = os.getenv("API_KEY")

#     MODEL_NAME_MAP = {
#         "gemini2.5": "gemini-2.5-flash-image-preview",
#         "gemini3": "gemini-3-pro-image-preview",
#         "gpt": "gpt-image-1.5",
#     }

#     model_name = MODEL_NAME_MAP[args.model]

#     asyncio.run(
#         generate_and_save_image(
#             prompt=args.prompt,
#             out_path=args.out,
#             model=model_name,
#             api_url=API_URL,
#             api_key=API_KEY,
#         )
#     )

# if __name__ == "__main__":
#     main()
