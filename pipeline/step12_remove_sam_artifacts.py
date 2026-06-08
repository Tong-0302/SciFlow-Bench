import os
import json
import argparse
import asyncio
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller
import re
def force_delete(desc: str) -> bool:
    """Return True if the node must be deleted regardless of VLM output."""
    if desc is None:
        return False

    s = desc.strip()

    # ① 单个字母必须删除（除非是 f_top / f_{front} 这种带后缀）
    if re.fullmatch(r"[A-Za-z]", s):
        return True
    
    if len(s) == 1 and not s.isalnum():
        return True

    # ③ 多字符但全部是符号，例如 "--", "···", "###"
    if all(not ch.isalnum() for ch in s):
        return True
    
    # ② 图编号 (a)(b)(c)
    if re.fullmatch(r"\([a-zA-Z]\)", s):
        return True

    return False

# ============================================================
# VLM Prompt（用于判断desc是否为单一字母或通用单词）
# ============================================================

PROMPT = """
You are a vision-language model specialized in understanding scientific or system diagrams.

Your task is to evaluate the provided "desc" fields of nodes and determine whether they should be deleted based on the following criteria:
1. If the "desc" contains terms that are widely used and generalized across multiple fields, without providing context-specific meaning to the diagram, then the node should be deleted.
2. If the "desc" contains terms that are more specialized, context-specific, or are abbreviations/acronyms that hold particular meaning within the specific diagram or domain, then the node should not be deleted.
3. Do not delete nodes with descriptions that are necessary for understanding the structure or behavior of the diagram, even if they are short or use abbreviations, as long as they have context-specific significance.

The nodes have the following structure:
- node_id: The unique identifier for the node.
- desc: The description text of the node.

Please evaluate each node and return a list of node IDs that should be deleted.

Output a strict JSON array with node IDs to delete:
[
  "node_id_1",
  "node_id_2",
  ...
]
"""

# ============================================================
# Utilities
# ============================================================

def load_jsonl(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: arr.append(json.loads(line))
            except: pass
    return arr


def extract_json_array(text):
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("No JSON array found in VLM response.")
    return json.loads(m.group(0))


# ============================================================
# VLM 调用 —— 仅对 desc 需要分析的节点
# ============================================================

async def ask_vlm_for_deletion(state, nodes):
    caller = VisionLLMCaller(
        state=state,
        vlm_config={"mode": "understanding"}  # 没有图像，仅分析desc
    )

    # 构建请求内容，输入desc进行分析
    content = PROMPT + "\nNODES:\n" + json.dumps(nodes, ensure_ascii=False)

    msg = await caller.call([HumanMessage(content=content)])
    raw = msg.content.strip()

    try:
        return extract_json_array(raw)
    except:
        print("❌ VLM JSON 解析失败。原始内容：")
        print(raw)
        return []


# ============================================================
# Main
# ============================================================


async def main(nodes_path, out_path):
    nodes = load_jsonl(nodes_path)

    # -------------------------------
    # Step 1: 强制删除（不进 VLM）
    # -------------------------------
    force_delete_ids = set()
    for n in nodes:
        desc = n.get("desc", "").strip()
        if desc and force_delete(desc):
            force_delete_ids.add(n["node_id"])

    print(f"⚠️ 强制删除 {len(force_delete_ids)} 个节点（单字母或(a)等图编号）")

    # -------------------------------
    # Step 2: VLM 删除逻辑（只处理未命中强制删除的）
    # -------------------------------
    nodes_for_vlm = [
        n for n in nodes
        if n["node_id"] not in force_delete_ids
        and n.get("desc", "").strip() != ""
    ]

    from types import SimpleNamespace
    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    print(f"🔍 调用 VLM 分析 {len(nodes_for_vlm)} 个节点...")

    # 调用 VLM 获取删除节点
    vlm_delete_ids = set(await ask_vlm_for_deletion(state, nodes_for_vlm))

    # -------------------------------
    # Step 3: 最终保留
    # -------------------------------
    final_nodes = [
        n for n in nodes
        if n["node_id"] not in force_delete_ids
        and n["node_id"] not in vlm_delete_ids
    ]

    # 保存结果
    with open(out_path, "w", encoding="utf-8") as f:
        for x in final_nodes:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("===================================")
    print(f"强制删除: {len(force_delete_ids)} 个")
    print(f"VLM 删除: {len(vlm_delete_ids)} 个")
    print(f"最终剩余: {len(final_nodes)} 个")
    print("输出文件:", out_path)
    print("===================================")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.nodes, args.out))
