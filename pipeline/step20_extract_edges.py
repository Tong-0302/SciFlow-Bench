import argparse
import asyncio
import json
import os
import re
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ===========================================================
# 环境变量
# ===========================================================
API_URL = os.getenv("DF_API_URL", "http://localhost:3888/v1")
API_KEY = os.getenv("DF_API_KEY","")
MODEL = os.getenv("DF_MODEL", "")   


# ===========================================================
# 通用加载
# ===========================================================
def load_nodes(path: str) -> List[Dict]:
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                nodes.append(json.loads(line))
    return nodes


def load_mermaid(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ===========================================================
# Edge Extraction Prompt（TEXT ONLY）
# ===========================================================
EDGE_PROMPT = """
You are a vision-language model analyzing a scientific diagram.

You are given:
1. A list of nodes (node_id, desc, bbox).
2. The Mermaid code, presented STRICTLY line-by-line.

Your task:
Resolve each Mermaid line “A --> B” into the correct pair of node_ids.

====================================================
ABSOLUTE OUTPUT RULES
====================================================

1. **Each Mermaid line corresponds to exactly ONE output edge.**
2. **If Mermaid contains duplicate lines (e.g., “A --> B”), you MUST output duplicate edges.**
3. **The number of output edges MUST equal the number of Mermaid lines.**
4. **You MUST NOT merge, condense, deduplicate, reinterpret, or optimize repeated lines.**
5. **Output edges MUST follow the SAME ORDER as the Mermaid lines (line 0, line 1, …).**
6. **Never skip a line. Never merge lines. Never combine multiple Mermaid lines.**

====================================================
NODE MATCHING RULES
====================================================

For each Mermaid line “SRC --> DST”:

• Match SRC and DST to nodes whose `desc` best corresponds to the label.
• If multiple nodes share the same label (“Conv”, “HFR”, etc.), choose the correct one
  by analyzing the overall diagram structure:

  - Arrows indicate spatial flow direction.
  - True 1→n branching targets appear in a clear horizontal or vertical cluster.
  - For duplicate labels, pick the node whose bbox position aligns with the branch.

• The diagram must be used to resolve ambiguities, but the Mermaid LINES define
  which edges MUST appear.

====================================================
OUTPUT STRICT JSON:
{
  "edges": [
    {"edge_id": "edge0", "from": "nodeX", "to": "nodeY"},
    {"edge_id": "edge1", "from": "nodeA", "to": "nodeB"}
  ]
}

"""

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def fix_broken_strings(text: str) -> str:
    """
    修复被 OCR / LLM 换行打断的字符串：
    "node\n15" -> "node15"
    """
    return re.sub(
        r'"([^"\n]*)\n([^"\n]*)"',
        lambda m: '"' + m.group(1).strip() + m.group(2).strip() + '"',
        text
    )


def extract_edges_loose(text: str) -> List[Dict]:
    """
    最终兜底：只用 regex 抽取 from / to
    """
    edges = []
    pairs = re.findall(
        r'"from"\s*:\s*"([^"]+)"\s*,\s*"to"\s*:\s*"([^"]+)"',
        text,
        flags=re.S
    )
    for i, (src, dst) in enumerate(pairs):
        edges.append({
            "edge_id": f"edge{i}",
            "from": src.strip(),
            "to": dst.strip()
        })
    return edges


def robust_parse_edges(content: str) -> Dict:
    """
    不管 LLM 返回什么，尽最大努力恢复 edges
    """
    raw = strip_code_fence(content)
    raw = fix_broken_strings(raw)

    # 1️⃣ 尝试直接 JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2️⃣ 尝试提取 {...}
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 3️⃣ 最终兜底：regex 抽 from / to
    edges = extract_edges_loose(raw)
    return {"edges": edges}


# ===========================================================
# LLM 调用
# ===========================================================
async def call_llm_edges(nodes: List[Dict], mermaid: str) -> Dict:
    node_text = "\n".join(
        [f"- {n['node_id']} | desc={n.get('desc','')} | bbox={n.get('bbox',[0,0,0,0])}"
         for n in nodes]
    )

    full_prompt = f"""
{EDGE_PROMPT}

Detected nodes:
---
{node_text}
---

Mermaid:
---
{mermaid}
---
"""

    llm = ChatOpenAI(
        openai_api_base=API_URL,
        openai_api_key=API_KEY,
        model_name=MODEL,
        temperature=0.2,
    )

    resp = await llm.ainvoke([HumanMessage(content=full_prompt)])
    content = resp.content or ""

    # ===== DEBUG：终端查看 RAW 输出 =====
    print("\n================ RAW LLM OUTPUT ================\n")
    print(content)
    print("\n================ END RAW OUTPUT ================\n")

    return robust_parse_edges(content)


# ===========================================================
# CLI 主逻辑
# ===========================================================
# async def main(nodes_path: str, mmd_path: str):
#     nodes = load_nodes(nodes_path)
#     mermaid = load_mermaid(mmd_path)

#     print("🚀 Extracting edges (TEXT ONLY MODE)…")

#     result = await call_llm_edges(nodes, mermaid)
#     edges = result.get("edges", [])

#     out_path = nodes_path.replace(".jsonl", "_edges.jsonl")
#     with open(out_path, "w", encoding="utf-8") as f:
#         for i, e in enumerate(edges):
#             f.write(json.dumps({
#                 "edge_id": f"edge{i}",
#                 "from": e["from"],
#                 "to": e["to"]
#             }, ensure_ascii=False) + "\n")

#     print(f"✅ Edges saved to: {out_path}")
#     print(json.dumps({"edges": edges}, indent=2, ensure_ascii=False))


# ===========================================================
# CLI
# # ===========================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--nodes", required=True)
#     parser.add_argument("--mmd", default="")
#     args = parser.parse_args()

#     asyncio.run(main(args.nodes, args.mmd))

async def main(nodes_path: str, mmd_path: str, output_path: str = None):
    nodes = load_nodes(nodes_path)
    mermaid = load_mermaid(mmd_path)

    print(f"🚀 Extracting edges (TEXT ONLY MODE) for {nodes_path}…")

    result = await call_llm_edges(nodes, mermaid)
    edges = result.get("edges", [])

    # [Fix] 如果没有指定 output_path，才使用默认的替换逻辑
    if not output_path:
        output_path = nodes_path.replace(".jsonl", "_edges.jsonl")

    # 确保父目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, e in enumerate(edges):
            f.write(json.dumps({
                "edge_id": f"edge{i}",
                "from": e.get("from", ""),
                "to": e.get("to", "")
            }, ensure_ascii=False) + "\n")

    print(f"✅ Edges saved to: {output_path}")
    print(json.dumps({"edges": edges}, indent=2, ensure_ascii=False))


# ===========================================================
# CLI
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--mmd", default="")
    # [Fix] 新增 output 参数
    parser.add_argument("--output", default=None, help="Path to save the output edges jsonl")
    
    args = parser.parse_args()

    asyncio.run(main(args.nodes, args.mmd, args.output))