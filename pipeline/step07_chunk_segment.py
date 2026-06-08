import os
import json
import argparse
import asyncio
import re
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller

# ============================================================
# JSON 解析（自动处理 ```json fenced block） 
# ============================================================
def extract_json_block(text):
    """
    自动检测并解析 VLM 输出中的 JSON，包括：
    - ```json ... ```
    - ``` ... ```
    - 原始 JSON
    """
    text = text.strip()

    # 尝试匹配 ```json ... ```
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()

    # 尝试匹配大括号包裹
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0).strip()

    return json.loads(text)


# ============================================================
# Prompt（通用性强、无示例、支持多标题）
# ============================================================
VLM_FIND_TITLES_PROMPT = """
You are a vision-language model specialized in identifying high-level functional
module titles in scientific or technical diagrams.

Your task:
Determine whether the given cropped diagram region contains a valid functional
module title.

A valid functional module title must satisfy all of the following:
• It explicitly names a high-level functional module or major component.
• It labels a relatively large region of the diagram, not an operation inside it.
• It is visually associated with a bounded region representing a complete module.
• It acts as a semantic identifier of a subsystem or major processing block.

The following must NOT be treated as functional module titles:
• small labels attached to icons, arrows, or individual operations
• repeated operator names that appear in multiple branches
• step descriptions, process phrases, or pipeline actions
• any local caption that does not clearly correspond to a complete module
• ambiguous labels that could be interpreted in multiple ways

Most importantly:
If there is any uncertainty about whether a text is a functional module title,
you must return an empty title list.

Output strict JSON:
{
  "titles": [
    {"title": "string"}
  ]
}

If no valid module title exists, or if uncertain, return:
{
  "titles": []
}
"""


# ============================================================
# 从单个 chunk 图像里提取标题
# ============================================================
async def extract_title_for_chunk(image_path):

    from types import SimpleNamespace
    vlm_state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    caller = VisionLLMCaller(
        state=vlm_state,
        vlm_config={"mode": "understanding", "input_image": image_path}
    )

    print(f"🔍 提取图像 {image_path} 的标题 …")
    msg = await caller.call([HumanMessage(content=VLM_FIND_TITLES_PROMPT)])
    raw = msg.content.strip()

    try:
        data = extract_json_block(raw)
        titles = [t["title"] for t in data.get("titles", [])]

        if not titles:
            return ""          # 没有标题 → 空字符串
        else:
            return "; ".join(titles)  # 多个标题 → 分号分隔

    except Exception as e:
        print(f"❌ JSON 解析失败：{raw}（错误：{e}）")
        return ""


# ============================================================
# 主过程：读取 chunks.jsonl，给每行加 title
# ============================================================
async def process_jsonl(input_jsonl, output_jsonl, image_folder):

    with open(input_jsonl, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line.strip()) for line in f.readlines()]

    updated_chunks = []

    for chunk in tqdm(chunks, desc="Step2: Describe nodes"):
        chunk_id = chunk["chunk_id"]
        image_path = os.path.join(image_folder, f"{chunk_id}.png")

        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在：{image_path}")
            chunk["title"] = ""  # 如果图像文件不存在，设置为空标题
        else:
            # 提取标题
            title = await extract_title_for_chunk(image_path)
            chunk["title"] = title  # 可为空，也可多个

        updated_chunks.append(chunk)

    # 保存新 JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for chunk in updated_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"📄 标题已添加 → {output_jsonl}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output", required=True, help="Path to save the updated JSONL file with titles")
    parser.add_argument("--img_folder", required=True, help="Path to the folder containing chunk images")
    args = parser.parse_args()

    asyncio.run(process_jsonl(args.input, args.output, args.img_folder))
