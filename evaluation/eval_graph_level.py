import json
import argparse
import os
import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. Data Loading
# ============================================================

def load_data(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Error] File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # JSON
    try:
        data = json.loads(content)
        if isinstance(data, list):
            data = data[0] if data else {}
        return {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", [])
        }
    except json.JSONDecodeError:
        pass

    # JSONL fallback
    nodes, edges = [], []
    for line in content.split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            nodes.extend(obj.get("nodes", []))
            edges.extend(obj.get("edges", []))
        except Exception:
            continue

    return {"nodes": nodes, "edges": edges}


# ============================================================
# 2. Embedding Model (Singleton, HF default behavior)
# ============================================================

_EMBED_MODEL = None

def load_embed_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Uses HuggingFace default cache behavior:
    - Model auto-downloads on first use
    - Cached in user's local HF cache (~/.cache/huggingface)
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"[Info] Loading embedding model: {model_name}")
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


# ============================================================
# 3. Hybrid Similarity (INLINE LOGIC)
# ============================================================

def hybrid_similarity_matrix(model, gt_nodes, pred_nodes):
    """
    Return sim matrix of shape [N_pred, N_gt]
    """
    gt_texts = [n.get("desc", "").strip() for n in gt_nodes]
    pred_texts = [n.get("desc", "").strip() for n in pred_nodes]

    N_gt, N_pred = len(gt_nodes), len(pred_nodes)
    sim = np.zeros((N_pred, N_gt), dtype=np.float32)

    if N_gt > 0 and N_pred > 0:
        gt_emb = model.encode(
            gt_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        pred_emb = model.encode(
            pred_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        sim = pred_emb @ gt_emb.T

    # Hybrid correction
    for i in range(N_pred):
        for j in range(N_gt):
            gt_empty = (gt_texts[j] == "")
            pred_empty = (pred_texts[i] == "")

            if gt_empty and pred_empty:
                gt_type = gt_nodes[j].get("node_type", "unknown")
                pred_type = pred_nodes[i].get("node_type", "unknown")
                sim[i, j] = 1.0 if gt_type == pred_type else 0.0

            elif gt_empty or pred_empty:
                sim[i, j] = 0.0

    return sim


# ============================================================
# 4. Node F1 (NO 1-to-1, Hybrid)
# ============================================================

def semantic_node_f1(
    model,
    gt_nodes,
    pred_nodes,
    sim_threshold=0.5
):
    if not gt_nodes:
        return (1.0, 1.0, 1.0) if not pred_nodes else (0.0, 0.0, 0.0)

    if not pred_nodes:
        return (0.0, 0.0, 0.0)

    sim = hybrid_similarity_matrix(model, gt_nodes, pred_nodes)

    # Precision: pred → gt
    pred_matched = (sim.max(axis=1) >= sim_threshold)
    tp_pred = pred_matched.sum()
    precision = tp_pred / (len(pred_nodes) + 1e-9)

    # Recall: gt → pred
    gt_matched = (sim.max(axis=0) >= sim_threshold)
    tp_gt = gt_matched.sum()
    recall = tp_gt / (len(gt_nodes) + 1e-9)

    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


# ============================================================
# 5. Edge F1 (NO node mapping, Hybrid)
# ============================================================

def semantic_edge_f1(
    model,
    gt_nodes,
    gt_edges,
    pred_nodes,
    pred_edges,
    sim_threshold=0.5
):
    """
    Relaxed, path-aware semantic edge F1.
    A predicted edge is correct if its endpoints
    semantically match the endpoints of ANY path in GT.
    """

    if not gt_edges:
        return (1.0, 1.0, 1.0) if not pred_edges else (0.0, 0.0, 0.0)

    # ---------- build node index ----------
    gt_id2idx = {n["node_id"]: i for i, n in enumerate(gt_nodes)}
    pred_id2idx = {n["node_id"]: i for i, n in enumerate(pred_nodes)}

    # ---------- similarity matrix ----------
    sim = hybrid_similarity_matrix(model, gt_nodes, pred_nodes)
    # sim shape: [N_pred, N_gt]

    # ---------- build GT reachability ----------
    from collections import defaultdict, deque

    adj = defaultdict(list)
    for e in gt_edges:
        u, v = e["from"], e["to"]
        if u in gt_id2idx and v in gt_id2idx:
            adj[u].append(v)
            adj[v].append(u) 

    reachable = defaultdict(set)

    for start in list(adj.keys()):
        queue = deque([start])
        visited = set([start])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    reachable[start].add(nxt)
                    queue.append(nxt)

    # ---------- edge matching ----------
    matched_gt_paths = set()
    tp = 0

    for pe in pred_edges:
        u_p, v_p = pe["from"], pe["to"]
        if u_p not in pred_id2idx or v_p not in pred_id2idx:
            continue

        pu, pv = pred_id2idx[u_p], pred_id2idx[v_p]

        # candidate GT start / end nodes
        cand_u = [
            gt_nodes[i]["node_id"]
            for i in range(len(gt_nodes))
            if sim[pu, i] >= sim_threshold
        ]
        cand_v = [
            gt_nodes[i]["node_id"]
            for i in range(len(gt_nodes))
            if sim[pv, i] >= sim_threshold
        ]

        matched = False
        for u_g in cand_u:
            for v_g in cand_v:
                if v_g in reachable.get(u_g, set()):
                    matched = True
                    matched_gt_paths.add((u_g, v_g))
                    break
            if matched:
                break

        if matched:
            tp += 1

    precision = tp / (len(pred_edges) + 1e-9)
    recall = len(matched_gt_paths) / (len(gt_edges) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1


# ============================================================
# 6. Main
# ============================================================

def evaluate(gt_path, pred_path, model_name="sentence-transformers/all-mpnet-base-v2"):
    print(f"\n[Evaluating]\n  GT:   {gt_path}\n  Pred: {pred_path}")

    gt_data = load_data(gt_path)
    pred_data = load_data(pred_path)
    model = load_embed_model(model_name)

    # Node
    n_p, n_r, n_f1 = semantic_node_f1(
        model,
        gt_data["nodes"],
        pred_data["nodes"]
    )
    print(f"  -> Node F1: {n_f1:.4f} (P={n_p:.3f}, R={n_r:.3f})")

    # Edge
    e_p, e_r, e_f1 = semantic_edge_f1(
        model,
        gt_data["nodes"],
        gt_data["edges"],
        pred_data["nodes"],
        pred_data["edges"]
    )
    print(f"  -> Edge F1: {e_f1:.4f} (P={e_p:.3f}, R={e_r:.3f})")

    final_score = 0.4 * n_f1 + 0.6 * e_f1

    # ✅ 完全保持你的原始返回结构
    return {
        "node_prec": n_p,
        "node_rec": n_r,
        "node_f1": n_f1,
        "edge_prec": e_p,
        "edge_rec": e_r,
        "edge_f1": e_f1,
        "final_score": final_score
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.gt, args.pred)



#  python -m dataflow_agent.bench.graph_level --gt autobench/cv_2025_autobench/2505.00752v2/canvas_full_merge.json --pred autobench/cv_2025_autobench/2505.00752v2/gemini3_autobench/canvas_full.json