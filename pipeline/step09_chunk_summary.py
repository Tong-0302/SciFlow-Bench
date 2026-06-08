import os
import json
import argparse
import asyncio
from tqdm import tqdm
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller


SUMMARY_PROMPT = """
You are a vision-language expert.

You are given a cropped chunk from a scientific/system diagram.
Your job is to generate a short, single-sentence natural-language summary
describing the main purpose or function of this chunk.

Use BOTH:
1. The content of the image.
2. The optional provided title (if any).

Rules:
- The summary must be a single concise sentence.
- Avoid hallucination: only describe what can be seen.
- If the title exists, incorporate it appropriately.
- If the image content is unclear, return an empty string "".

Output STRICT JSON only:
{
  "summary": "..."
}
"""


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except:
                pass
    return chunks


async def summarize_chunk(state, img_path, title):
    # Combine prompt & title
    full_prompt = SUMMARY_PROMPT
    if title:
        full_prompt += f'\nProvided title: "{title}"\n'

    caller = VisionLLMCaller(
        state=state,
        vlm_config={"mode": "understanding", "input_image": img_path}
    )

    msg = await caller.call([HumanMessage(content=full_prompt)])
    raw = msg.content.strip()

    # strict parsing
    try:
        return json.loads(raw)
    except:
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))

    return {"summary": ""}


async def main(chunks_path, imgs_dir, out_path):
    # EXACT SAME state pattern as step2_describe_nodes.py
    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    chunks = load_chunks(chunks_path)
    result = []

    for item in tqdm(chunks, desc="StepX: Summarize chunks"):
        cid = item["chunk_id"]
        title = item.get("title", "")
        img_path = os.path.join(imgs_dir, f"{cid}.png")

        if not os.path.exists(img_path):
            print(f"⚠️ Missing chunk image: {img_path}")
            item["summary"] = ""
            result.append(item)
            continue

        summary = await summarize_chunk(state, img_path, title)
        item["summary"] = summary.get("summary", "")

        result.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        for line in result:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("====================================")
    print("✨ Finish: chunk summaries generated.")
    print(f"Input JSONL : {chunks_path}")
    print(f"Images Dir  : {imgs_dir}")
    print(f"Output JSONL: {out_path}")
    print("====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--imgs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.chunks, args.imgs, args.out))
