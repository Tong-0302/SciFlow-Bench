import argparse
import json
import os
import ast
import asyncio
import re
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv() 

# ============================================================
# Embedding Model Cache (CRITICAL FIX)
# ============================================================
_EMBED_MODEL = None
def get_embed_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"[Info] Loading embedding model: {model_name}")
        _EMBED_MODEL = SentenceTransformer(
            model_name
            # ✅ no cache_folder here; HF default cache
        )
    return _EMBED_MODEL

# ============================================================
# 环境配置
# ============================================================
API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
API_KEY = os.getenv("DF_API_KEY")
MODEL = os.getenv("DF_MODEL", "gpt-4o")

# ============================================================
# 数据处理 Utils
# ============================================================
def load_graph_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prompt"]

def simplify_graph_for_llm(graph_json: Dict) -> Dict:
    nodes = []
    for n in graph_json.get("nodes", []):
        text = n.get("desc", "").strip() or n.get("vlm_prompt", "").strip() or n.get("node_type", "")
        nodes.append({"id": n.get("node_id"), "text": text})

    edges = []
    for e in graph_json.get("edges", []):
        edges.append({"from": e.get("from"), "to": e.get("to")})

    return {
        "nodes": nodes,
        "edges": edges
    }


def extract_node_descs(graph_json: Dict) -> List[str]:
    """
    AutoBench Text-Level Filtering (Recall-friendly)
    Goal: remove OCR / symbol garbage, keep anything that could be a module
    """
    descs = []

    for n in graph_json.get("nodes", []):
        text = (
            n.get("desc", "").strip()
            or n.get("vlm_prompt", "").strip()
            or n.get("node_type", "")
        )

        if not text:
            continue

        # normalize
        text = text.strip()

        # ---------- 1. Hard noise ----------
        # single char
        if len(text) == 1:
            continue

        # pure number
        if text.isdigit():
            continue

        # pure symbol
        if all(not c.isalnum() for c in text):
            continue

        # subfigure / index labels: (a), (1), (iii)
        if re.fullmatch(r"\([a-zA-Z0-9]+\)", text):
            continue

        # variable-like: x1, y2, z_k
        if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*\d+", text):
            continue

        # ---------- 2. Length sanity ----------
        # extremely long OCR garbage (optional)
        if len(text) > 80:
            continue

        # ---------- 3. Keep everything else ----------
        descs.append(text)

    return descs



# ============================================================
# LLM #1 —— 核心组件抽取
# ============================================================
async def extract_core_components_llm(method_text: str) -> List[str]:
    safe_text = method_text.replace("{", "{{").replace("}", "}}")
    
    prompt = f"""
You are a scientific-method analysis agent.
Given the method section below, extract 3–10 core components of the model.

Requirements:
- Output ONLY a JSON list of strings.
- Each component should be a functional module (e.g., "Encoder", "MLP", "Loss Function").
- Exclude generic terms like "Training" or "Dataset".
- Exclude mathematical symbols (e.g., "x", "y") unless they represent a major module.

Method Text:
----------------
{safe_text}
----------------

Output ONLY: ["Component A", "Component B"]
Do not output any explanation.
"""

    try:
        llm = ChatOpenAI(
            openai_api_base=API_URL, openai_api_key=API_KEY,
            model_name=MODEL, temperature=0.0
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        # 1. 清洗 Markdown
        clean_content = content.strip().strip("`")
        if clean_content.startswith("json"):
            clean_content = clean_content[4:].strip()
            
        # 2. 尝试标准 JSON 解析
        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            pass # 继续尝试兜底方案

        # 3. 正则提取 + AST 兜底 (专门处理末尾逗号 ["A", "B",])
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            list_str = match.group(0)
            try:
                # ast.literal_eval 可以处理 python 风格的 list (允许末尾逗号)
                parsed = ast.literal_eval(list_str)
                if isinstance(parsed, list):
                    return [str(i) for i in parsed] # 确保全是字符串
            except:
                # 如果 AST 也不行，尝试宽松的 JSON 修复 (比如把 ' 换成 " )
                try:
                    fixed_str = list_str.replace("'", '"')
                    return json.loads(fixed_str)
                except:
                    pass
                
    except Exception as e:
        print(f"[LLM Error] Extract components failed: {e}")
        
    return []

# ============================================================
# Coverage & Faithfulness (Embedding Based)
# ============================================================
def compute_coverage_faithfulness(core_components: List[str], node_descs: List[str]):
    # 保护：如果过滤完短文本后没剩几个节点了，也要处理
    if not core_components: return 0.0, 0.0
    if not node_descs: return 0.0, 0.0 

    model = get_embed_model()

    comp_emb = model.encode(core_components, normalize_embeddings=True)
    node_emb = model.encode(node_descs, normalize_embeddings=True)

    sims = comp_emb @ node_emb.T  # [C, N]

    # Threshold: 0.65
    THRESH = 0.65

    # Coverage
    covered_count = np.sum(sims.max(axis=1) > THRESH)
    coverage = covered_count / len(core_components)

    # Faithfulness
    supported_count = np.sum(sims.max(axis=0) > THRESH)
    hallucination_ratio = 1.0 - (supported_count / len(node_descs))
    faithfulness = 1.0 - hallucination_ratio

    return float(coverage), float(faithfulness)

# ============================================================
# LLM #2 —— 全图语义判别 (LLM Judge)
# ============================================================
async def llm_judge_alignment(method_text: str, graph_json: Dict) -> float:
    
    lean_graph = simplify_graph_for_llm(graph_json)

    graph_str = json.dumps(lean_graph, indent=2).replace("{", "{{").replace("}", "}}")
    safe_method = method_text.replace("{", "{{").replace("}", "}}")

    prompt = f"""
You are a strict judge evaluating a scientific diagram generated from a method description.

[Method Description]
{safe_method}

[Generated Diagram Structure (JSON)]
{graph_str}

Task: Rate the semantic consistency between the diagram and the text on a scale of 1 to 5.

Evaluation Criteria:
1. **Module Presence**: Do the diagram nodes reflect the major functional modules described in the text?
2. **Flow**: Does the edge connection logic match the method's pipeline?

Scale:
1: Completely irrelevant or wrong.
2: Poor alignment, missing major components.
3: Fair, captures basic structure but has errors.
4: Good, mostly correct.
5: Excellent, perfectly reflects the method logic.

Output ONLY the integer score (1, 2, 3, 4, or 5).
"""

    try:
        llm = ChatOpenAI(
            openai_api_base=API_URL,
            openai_api_key=API_KEY,
            model_name=MODEL,
            temperature=0.0
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        content = resp.content.strip().strip("`")

        match = re.search(r"\b[1-5]\b", content)
        if match:
            score_1to5 = int(match.group(0))
            return (score_1to5 - 1) / 4.0
    except Exception as e:
        print(f"[LLM Error] Judge failed: {e}")

    return 0.0


# ============================================================
# Pipeline
# ============================================================
async def evaluate(method_path, graph_path):
    method_text = load_prompt_text(method_path)
    graph_json = load_graph_json(graph_path)

    # 1. 抽取核心组件
    core_components = await extract_core_components_llm(method_text)
    
    # 2. Embedding 指标 (使用过滤后的 node_descs)
    node_descs = extract_node_descs(graph_json)
    coverage, faithfulness = compute_coverage_faithfulness(core_components, node_descs)

    # 3. LLM Judge (使用包含 Chunks 的 JSON)
    alignment_score = await llm_judge_alignment(method_text, graph_json)

    return {
        "metrics": {
            "coverage": coverage,
            "faithfulness": faithfulness,
            "llm_alignment_score": alignment_score,
            "final_text_score": 0.3*coverage + 0.3*faithfulness + 0.4*alignment_score
        },
        "details": {
            "extracted_components": core_components,
            "valid_node_count": len(node_descs),
            "original_node_count": len(graph_json.get("nodes", []))
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--graph", required=True)
    args = parser.parse_args()
    
    res = asyncio.run(evaluate(args.text, args.graph))
    print(json.dumps(res, indent=2, ensure_ascii=False))




# python -m dataflow_agent.bench.text_level --text autobench/cv_2025_autobench/2505.00752v2/prompt.json --graph autobench/cv_2025_autobench/2505.00752v2/gemini3_autobench/canvas_full.json