import os
import json
import argparse
import asyncio
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller
from langchain_openai import ChatOpenAI

# ============================================================
# VLM Prompt（用于判断desc是否为专有术语，删除专有术语）
# ============================================================

DELETE_PROMPT = """
You are a vision-language model specialized in understanding scientific or system diagrams.

Your task is to evaluate the provided "desc" fields of nodes and determine whether they should be deleted based on the following criteria:
1. If the "desc" contains terms that are widely used and generalized across multiple fields, without providing context-specific meaning to the diagram, then the node should **not** be deleted. These terms are often common in many areas and provide general context to the diagram.
2. If the "desc" contains terms that are specialized, context-specific, or are abbreviations/acronyms that hold particular meaning within the specific diagram or domain, then the node **should** be deleted. These terms are often domain-specific and don't add general context to the diagram.
3. Do not delete nodes with descriptions that are necessary for understanding the structure or behavior of the diagram, even if they are short or use abbreviations, as long as they are context-specific and meaningful within the diagram's domain.

The nodes have the following structure:
- desc: The description text of the node.
- bbox: The bounding box of the node.

Please evaluate each node and return a list of descriptions ("desc") that should be deleted.

Output a strict JSON array with descriptions to delete:
[
  "desc_1",
  "desc_2",
  ...
]
"""
NODE_TYPE_PROMPT = """
You are an expert ML system for classifying OCR text nodes from scientific diagrams.

Given ONLY the extracted OCR text (desc), classify its node_type.

Valid types:
- "text_block": textual content (title, paragraph, label, etc.)
- "shape": module box, process box, arrow, geometric block, etc.
- "image_placeholder": mini image/icon that represents a picture or figure.
- "table_placeholder": content visually resembling tables or grids.
- "annotation": comment-like or explanatory visual elements.

Rules:
- Always return exactly one node_type.
- Use only the text itself — no image is provided.
- If unsure, default to "text_block".

Output strict JSON:
{
  "node_type": "..."
}
"""

# ============================================================
# Utilities
# ============================================================

def load_jsonl(path):
    """加载JSONL文件"""
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                arr.append(json.loads(line))
            except Exception as e:
                print(f"Error loading line: {line}")
                print(e)
    return arr

def extract_json_array(text):
    """从VLM返回的文本中提取JSON数组"""
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("No JSON array found in VLM response.")
    return json.loads(m.group(0))

def extract_json_object(text):
    import re, json
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except:
        return {}
    
# ============================================================
# VLM 调用 —— 仅对 desc 需要分析的节点
# ============================================================

async def ask_vlm_for_deletion(state, nodes):
    caller = VisionLLMCaller(
        state=state,
        vlm_config={"mode": "understanding"}  # 没有图像，仅分析desc
    )

    # 构建请求内容，输入desc进行分析
    content = DELETE_PROMPT + "\nNODES:\n" + json.dumps(nodes, ensure_ascii=False)

    msg = await caller.call([HumanMessage(content=content)])
    raw = msg.content.strip()

    try:
        # 直接从 VLM 返回的原始内容中提取删除的desc字段列表
        result = extract_json_array(raw)
        return result  # 返回的是desc字段列表
    except Exception as e:
        print(f"❌ VLM JSON 解析失败。原始内容：{raw}")
        print(e)
        return []

async def classify_node_type(state, desc: str) -> str:
    if not desc.strip():
        return "text_block"

    llm = ChatOpenAI(
        openai_api_base=state.request.chat_api_url,
        openai_api_key=state.request.api_key,
        model_name=state.request.model,
        temperature=0.0,
    )

    msg = [HumanMessage(content=NODE_TYPE_PROMPT + f'\nDESC: "{desc}"')]
    resp = await llm.ainvoke(msg)

    raw = resp.content.strip()
    obj = extract_json_object(raw)
    return obj.get("node_type", "text_block")

# ============================================================
# Main
# ============================================================

async def main(nodes_path, out_path):
    # 读取输入
    nodes = load_jsonl(nodes_path)

    # 仅选择 desc 非空的节点进行 VLM 分析
    nodes_with_desc = [
        n for n in nodes
        if n.get("desc", "").strip() != ""
    ]

    # 构建 state
    from types import SimpleNamespace
    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    print(f"🔍 调用 VLM，分析 {len(nodes_with_desc)} 个节点的描述内容...")

    # 获取 VLM 判断的删除desc
    desc_to_delete = await ask_vlm_for_deletion(state, nodes_with_desc)

    # 保留那些未被删除的节点，删除 desc 字段匹配的节点
    final_nodes = [
        n for n in nodes
        if n["desc"] not in desc_to_delete  # 仅根据desc判断删除
    ]

    # 保存输出
    with open(out_path, "w", encoding="utf-8") as f:
        for x in final_nodes:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"✅ 过滤后的节点已保存 → {out_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.nodes, args.out))

