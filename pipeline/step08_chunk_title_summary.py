import os
import json
import argparse
import asyncio
import re
from tqdm import tqdm
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller


# ============================================================
# JSON 自动解析（支持 ```json fenced block）
# ============================================================
def extract_json_block(text):
    text = text.strip()

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()

    # { ... }
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0).strip()

    return json.loads(text)


# ============================================================
# 统一 Prompt（包含 title detection + summary）
# ============================================================
CHUNK_UNIFIED_PROMPT = """
You are a vision-language model specialized in analyzing scientific or technical
diagram chunks.

You are given:
1. The cropped chunk image.
2. (Optional) The module title text already extracted from the paper, if any.

Your task:
• Determine whether the cropped region contains any valid functional module titles.
• Produce a single-sentence natural-language summary of the chunk’s function.


-------------------------------------
TITLE EXTRACTION RULES (UNIVERSAL)
-------------------------------------
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


-------------------------------------
SUMMARY RULES
-------------------------------------
• The summary MUST be exactly one concise sentence.
• It MUST be based solely on the visual information contained in the chunk.
• It MUST NOT depend on, restate, or reference the module title text.
• It MUST NOT infer unseen or speculative functionality.
• If the visual content is unclear, the summary must be empty.


-------------------------------------
OUTPUT FORMAT (STRICT JSON)
-------------------------------------
You MUST output valid JSON only:

{
  "titles": [
    {"title": "string"}
  ],
  "summary": "string"
}

• If no titles are found, return:
  "titles": []
• Do NOT include comments, explanations, or extra text.
"""


# ============================================================
# 单 chunk：一次调用 VLM → 输出 title + summary
# ============================================================
async def analyze_chunk(state, img_path):
    caller = VisionLLMCaller(
        state=state,
        vlm_config={"mode": "understanding", "input_image": img_path}
    )

    msg = await caller.call([HumanMessage(content=CHUNK_UNIFIED_PROMPT)])
    raw = msg.content.strip()

    try:
        return extract_json_block(raw)
    except Exception:
        print(f"❌ JSON 解析失败：{raw}")
        return {"titles": [], "summary": ""}


# ============================================================
# 主函数
# ============================================================
async def main(chunks_path, imgs_dir, out_path):

    # VLM 状态
    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except:
                pass

    result = []

    for item in tqdm(chunks, desc="Unified Title + Summary"):
        cid = item["chunk_id"]
        img_path = os.path.join(imgs_dir, f"{cid}.png")

        if not os.path.exists(img_path):
            print(f"⚠ Missing chunk image: {img_path}")
            item["titles"] = []
            item["summary"] = ""
            result.append(item)
            continue

        data = await analyze_chunk(state, img_path)

        # 写入结果
        titles_list = [t["title"] for t in data.get("titles", [])]
        if len(titles_list) == 0:
            item["titles"] = ""
        else:
            item["titles"] = "; ".join(titles_list)

        item["summary"] = data.get("summary", "")

        result.append(item)

    # 输出 JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for line in result:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("====================================")
    print("✨ DONE: titles + summary generated!")
    print(f"Input  : {chunks_path}")
    print(f"Images : {imgs_dir}")
    print(f"Output : {out_path}")
    print("====================================")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--imgs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.chunks, args.imgs, args.out))
