import os
import json
import argparse
import asyncio
import re
from tqdm import tqdm
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller


def force_delete(desc: str) -> bool:
    """Return True if the node must be deleted regardless of VLM output."""
    if desc is None:
        return False

    s = desc.strip()

    # ① 单个字母必须删除（除非是 f_top / f_{front} 这种带后缀）
    if re.fullmatch(r"[A-Za-z]", s):
        return True

    # ② 图编号 (a)(b)(c)
    if re.fullmatch(r"\([a-zA-Z]\)", s):
        return True

    return False

# ============================================
# JSON extractor
# ============================================
def extract_json_block(text):
    text = text.strip()

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()

    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return json.loads(m.group(0))

    return json.loads(text)


# ============================================
# Unified Prompt：一次完成 desc 删除 + title 删除
# ============================================
UNIFIED_PROMPT = """
You are a vision-language model analyzing a node inside a scientific or
technical diagram.

You are given:
1. The full chunk image where this node appears.   (Only used for title detection)
2. The chunk’s module title text (may be empty).
3. The node’s information:
   • desc
   • vlm_prompt
   • bbox

You must output TWO FULLY INDEPENDENT decisions:



=====================================================================
(1) DESC-BASED DELETION  — STRICT GENERAL RULES (IMAGE NOT USED)
=====================================================================
This decision must rely **only on the desc text**.

A node should be deleted only when ALL the following conditions hold:

• The desc is short.
• The desc does not denote any operation, module, function, variable,
  tensor, data entity, or any component that directly participates in
  the system’s computational or logical structure.
• The desc represents only auxiliary, decorative, or non-structural text,
  such as labels, axis-like words, marginal annotations, category names,
  standalone markers, or any text not involved in the system’s function.
• Removing the node does not affect understanding of the system’s
  structure, flow, or semantics.

A node must be kept whenever ANY of the following is true:

• The desc contributes functional, computational, or structural meaning.
• The desc corresponds to any component that interacts with the flow.
• The desc denotes anything that is part of the diagram’s logical system.

OUTPUT → "delete_desc": true or false



=====================================================================
(2) TITLE-BASED DETECTION — STRICT GENERAL RULES
=====================================================================
This decision may use both text and visual cues.

A node must be classified as a title node when BOTH of the following
conditions are strongly satisfied:

(A) Textual relation:
    The desc matches, includes, forms a meaningful part of, or directly
    corresponds to the semantic content of the chunk’s module title.
    Partial matches are valid only when they convey genuine title meaning,
    not incidental similarity.

(B) Spatial or visual consistency:
    The node appears in a region where titles typically occur, or its
    visual form is consistent with titles, such as plain text labeling,
    header-like appearance, or other typical title characteristics.

If (A) is strongly satisfied and (B) is not contradicted, classify as title.

If (A) is weak or absent, classify as non-title.

OUTPUT → "is_title_node": true or false



=====================================================================
STRICT JSON OUTPUT ONLY:
{
  "delete_desc": true or false,
  "is_title_node": true or false
}
"""


# ============================================
# Call VLM for one node
# ============================================
async def analyze_node(state, chunk_img, chunk_title, node):

    caller = VisionLLMCaller(
        state=state,
        vlm_config={"mode": "understanding", "input_image": chunk_img}
    )

    prompt = (
        UNIFIED_PROMPT
        + "\n\nChunk title(s): " + str(chunk_title)
        + "\nNode info:\n" + json.dumps({
            "desc": node.get("desc", ""),
            "vlm_prompt": node.get("vlm_prompt", ""),
            "bbox": node.get("bbox", "")
        }, ensure_ascii=False)
    )

    msg = await caller.call([HumanMessage(content=prompt)])
    raw = msg.content.strip()

    try:
        data = extract_json_block(raw)
        delete_desc = bool(data.get("delete_desc", False))
        is_title_node = bool(data.get("is_title_node", False))
        return delete_desc, is_title_node

    except Exception:
        print("⚠ JSON parse failed:", raw)
        return False, False


# ============================================
# Main
# ============================================
async def main(chunks_jsonl, nodes_jsonl, chunk_img_dir, out_jsonl):

    # 加载 chunk titles
    chunk_titles = {}
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_titles[c["chunk_id"]] = c.get("title", "")

    # 加载节点
    nodes = [json.loads(line) for line in open(nodes_jsonl, "r", encoding="utf-8")]

    # VLM 状态
    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    final_nodes = []

    for node in tqdm(nodes, desc="Unified desc-delete + title-delete"):

        cid = node["chunk_id"]
        chunk_title = chunk_titles.get(cid, "")
        chunk_img = os.path.join(chunk_img_dir, f"{cid}.png")

        if not node.get("desc", "").strip():
            delete_desc = False
            is_title_node = False
            final_nodes.append(node)
            continue

        if not os.path.exists(chunk_img):
            final_nodes.append(node)
            continue

        if force_delete(node.get("desc", "")):
            print(f"🗑 强制删除 {node['node_id']}（硬规则）")
            continue

        delete_desc, is_title_node = await analyze_node(state, chunk_img, chunk_title, node)

        # 删除规则
        if delete_desc:
            print(f"🗑 删除节点 {node['node_id']}（desc 过于通用）")
            continue

        if is_title_node:
            print(f"🗑 删除节点 {node['node_id']}（属于 chunk title）")
            continue

        final_nodes.append(node)

    # 输出结果
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for n in final_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    print("====================================")
    print("✨ Unified deletion completed!")
    print("Output:", out_jsonl)
    print("====================================")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--chunk_imgs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asyncio.run(main(
        args.chunks, args.nodes, args.chunk_imgs, args.out
    ))
