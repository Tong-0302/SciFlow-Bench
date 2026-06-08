import os
import json
import argparse
import asyncio
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller


LAYOUT_PROMPT = """
You are a vision-language model analyzing the global layout of a scientific or system diagram.

Your job is to determine the PRIMARY data-flow or layout direction of the entire diagram.

Possible values:
- "left-to-right"
- "right-to-left"
- "top-to-bottom"
- "bottom-to-top"
- "grid"          (if layout is a symmetric grid or table-like)
- "other"         (if the layout does not follow a clear directional flow)

Rules:
- Only output ONE direction.
- Do NOT describe content.
- Focus ONLY on the major arrangement of blocks/arrows.
- If arrows conflict, pick the globally dominant direction.

Output STRICT JSON only:
{
  "layout_flow": "..."
}
"""


async def detect_layout(state, img_path):
    caller = VisionLLMCaller(
        state=state,
        vlm_config={"mode": "understanding", "input_image": img_path}
    )

    msg = await caller.call([HumanMessage(content=LAYOUT_PROMPT)])
    raw = msg.content.strip()

    # First try direct JSON parsing
    try:
        return json.loads(raw)
    except:
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))

    return {"layout_flow": "other"}


async def main(img_path, out_path):
    # Construct state EXACTLY like your working code
    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    result = await detect_layout(state, img_path)

    # Write one-line JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("====================================")
    print("✨ Layout detection finished")
    print(f"Image   : {img_path}")
    print(f"Output  : {out_path}")
    print(f"Result  : {result}")
    print("====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.img, args.out))
