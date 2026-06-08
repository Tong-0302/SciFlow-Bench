import os
import json
import argparse
import asyncio
import re
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller

# ============================================================
# 统一 PROMPT：一次返回所有字段
# ============================================================
UNIFIED_PROMPT = """
You are a vision-language model analyzing a cropped node from a scientific or system diagram.

Your output MUST contain EXACTLY FOUR fields and they MUST be independent:
"desc", "vlm_prompt", "node_type", "meaningful".

----------------------------------------------------
1. desc (text inside the node)
----------------------------------------------------
• Extract readable text inside the node.
• Return "" if no readable text exists.
• MUST NOT return placeholders such as "...", "N/A", "none", "unknown".

----------------------------------------------------
2. vlm_prompt (visual appearance description)
----------------------------------------------------
Provide a precise and neutral visual description including:
• shape: rectangle, rounded-rectangle, arrow, icon, block, bar, etc.
• internal structure: stacked layers, symbols, arrows, grids, repeated shapes
• colors
• spatial layout: vertical block, horizontal block, small icon on the left, etc.

MUST NOT mention meaning or functionality.
MUST NOT say “this node is meaningful/meaningless”.

----------------------------------------------------
3. node_type (choose EXACTLY ONE)
----------------------------------------------------
• "text_block"          – main content is text
• "shape"               – module box, simple arrow, geometric block
• "image_placeholder"   – miniature photo/figure/icon
• "table_placeholder"   – grid/table-like visuals
• "annotation"          – comment-like or explanatory decorations

Choose based ONLY on visual form, NOT meaning.

----------------------------------------------------
4. meaningful (true/false)
----------------------------------------------------
A node is MEANINGLESS if ANY of the following is true:
• It contains only a single simple element with no internal structure.
• It is purely decorative, stylistic, or isolated without functional clues.
• It has low informational content (uniform region, simple geometric patch).
• It is a standalone arrow shape (→, ↑, ↓, ↗) with no internal structure.
  (IMPORTANT: Any node whose entire content is just one arrow MUST be meaningless.)
• It is a small standalone icon without structural detail.

A node is MEANINGFUL if ANY is true:
• It contains multiple internal components or layered structures.
• It shows combined shapes that imply processing, grouping, hierarchy, or flows.
• It contributes to expressing relationships or data movement BETWEEN modules.
  (NOT just a single arrow shape; must have structural context.)

Return strictly:
true  or  false

----------------------------------------------------
Output STRICT JSON ONLY:
{
  "desc": "...",
  "vlm_prompt": "...",
  "node_type": "...",
  "meaningful": true or false
}
"""


# ============================================================
# JSON extractor
# ============================================================
def extract_json_block(text):
    text = text.strip()

    # Try ```json``` fenced block
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()

    # Try {...}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0).strip()

    return json.loads(text)


# ============================================================
# 单节点：一次 VLM
# ============================================================
async def analyze_node(state, img_path):
    caller = VisionLLMCaller(
        state=state,
        vlm_config={"mode": "understanding", "input_image": img_path}
    )
    msg = await caller.call([HumanMessage(content=UNIFIED_PROMPT)])
    raw = msg.content.strip()

    try:
        return extract_json_block(raw)
    except:
        print(f"❌ JSON parse failed: {raw}")
        return {
            "desc": "",
            "vlm_prompt": "",
            "node_type": "shape",
            "meaningful": True
        }


def load_nodes(path):
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                nodes.append(json.loads(line))
            except:
                pass
    return nodes


# ============================================================
# 主逻辑
# ============================================================
async def main(nodes_path, imgs_dir, out_path):

    from types import SimpleNamespace

    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    nodes = load_nodes(nodes_path)
    final_nodes = []

    for node in tqdm(nodes, desc="Step4 (Unified describe + filter meaningless)"):
        nid = node["node_id"]
        img_path = os.path.join(imgs_dir, f"{nid}.png")

        if not os.path.exists(img_path):
            print(f"⚠ Missing image: {img_path}")
            continue

        data = await analyze_node(state, img_path)

        node["desc"] = data.get("desc", "")
        node["vlm_prompt"] = data.get("vlm_prompt", "")
        node["node_type"] = data.get("node_type", "shape")
        meaningful = data.get("meaningful", True)

        # desc 有内容 → 必保留
        if node["desc"].strip() != "":
            final_nodes.append(node)
            continue

        # desc 空 → 看 meaningful
        if not meaningful:
            print(f"🗑 Deleted meaningless node: {nid}")
            continue

        # meaningful = True → 保留
        final_nodes.append(node)

    # 输出
    with open(out_path, "w", encoding="utf-8") as f:
        for n in final_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    print("====================================================")
    print("✨ Finished Step4 (Unified VLM)")
    print(f"Output saved to: {out_path}")
    print(f"Nodes kept: {len(final_nodes)}")
    print("====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--imgs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.nodes, args.imgs, args.out))


