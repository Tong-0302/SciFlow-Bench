# import json
# import argparse
# from pathlib import Path


# def load_jsonl(path):
#     items = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 items.append(json.loads(line))
#     return items


# def load_edges_file(path):
#     """兼容两种格式：JSON 或 JSONL"""
#     txt = Path(path).read_text(encoding="utf-8").strip()
#     if not txt:
#         return []

#     # 可能是一整个 JSON { "edges": [...] }
#     try:
#         obj = json.loads(txt)
#         if isinstance(obj, dict) and "edges" in obj:
#             return obj["edges"]
#         if isinstance(obj, list):
#             return obj
#     except:
#         pass

#     # fallback: JSONL
#     items = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 items.append(json.loads(line))
#     return items


# def main(out_dir, output_json):
#     out_dir = Path(out_dir)
#     print("🔧 Step13 assembling global JSON…")

#     # ------------------------------------------------
#     # Load canvas & layout
#     # ------------------------------------------------
#     layout = load_jsonl(out_dir / "layout.jsonl")[0]
#     canvas = load_jsonl(out_dir / "canvas.jsonl")[0]

#     # ------------------------------------------------
#     # Load chunks
#     # ------------------------------------------------
#     chunks_path = out_dir / "chunks" / "chunks_with_titles_and_summary.jsonl"
#     if not chunks_path.exists():
#         raise FileNotFoundError(f"Missing {chunks_path}")

#     chunk_items = load_jsonl(chunks_path)

#     # chunk_id → bbox
#     chunk_bbox_map = {}
#     final_chunks = []

#     for c in chunk_items:
#         cid = c["chunk_id"]
#         bbox = c["bbox"]
#         chunk_bbox_map[cid] = {
#             "x": bbox[0],
#             "y": bbox[1],
#             "w": bbox[2],
#             "h": bbox[3],
#         }
#         final_chunks.append({
#             "chunk_id": cid,
#             "title": c.get("title", ""),
#             "summary": c.get("summary", ""),
#             "node_ids": []
#         })

#     # ------------------------------------------------
#     # Load nodes from nodes_work/*/nodes_final_local.jsonl
#     # 重新编号 node0, node1, ...
#     # ------------------------------------------------
#     nodes_work_dir = out_dir / "nodes_work"
#     if not nodes_work_dir.exists():
#         raise FileNotFoundError(f"Missing {nodes_work_dir}")

#     node_id_map = {}             # (chunk_id, old_node_id) -> new_node_id
#     final_nodes = []
#     positions_nodes = []
#     chunk_to_node_ids = {c["chunk_id"]: [] for c in final_chunks}

#     global_node_idx = 0

#     for chunk_dir in sorted(nodes_work_dir.iterdir()):
#         if not chunk_dir.is_dir():
#             continue

#         chunk_id = chunk_dir.name
#         local_nodes_path = chunk_dir / "nodes_final_local.jsonl"
#         if not local_nodes_path.exists():
#             continue

#         local_nodes = load_jsonl(local_nodes_path)
#         chunk_bbox = chunk_bbox_map[chunk_id]

#         for n in local_nodes:
#             old_id = n["node_id"]
#             new_id = f"node{global_node_idx}"
#             global_node_idx += 1

#             node_id_map[(chunk_id, old_id)] = new_id
#             chunk_to_node_ids[chunk_id].append(new_id)

#             # 转换 bbox 为全局坐标
#             lx, ly, w, h = n["bbox"]
#             global_bbox = {
#                 "x": chunk_bbox["x"] + lx,
#                 "y": chunk_bbox["y"] + ly,
#                 "w": w,
#                 "h": h,
#             }

#             positions_nodes.append({
#                 "node_id": new_id,
#                 "bbox": global_bbox
#             })

#             final_nodes.append({
#                 "node_id": new_id,
#                 "chunk_id": chunk_id,
#                 "node_type": n.get("node_type", ""),
#                 "desc": n.get("desc", ""),
#                 "render_method": n.get("render_method", ""),
#                 "vlm_prompt": n.get("vlm_prompt", "")
#             })

#     # 写入 final_chunks[*].node_ids
#     for c in final_chunks:
#         c["node_ids"] = chunk_to_node_ids[c["chunk_id"]]

#     # ------------------------------------------------
#     # Load edges & remap node_id
#     # ------------------------------------------------
#     final_edges = []
#     global_edge_idx = 0

#     for chunk_dir in sorted((out_dir / "nodes_work").iterdir()):
#         if not chunk_dir.is_dir():
#             continue

#         chunk_id = chunk_dir.name

#         edge_file = chunk_dir / "nodes_final_local_edges.jsonl"
#         if not edge_file.exists():
#             print(f"⚠ {chunk_id}: no edges file {edge_file}, skip")
#             continue

#         edges = load_edges_file(edge_file)

#         for e in edges:
#             old_from = e.get("from")
#             old_to = e.get("to")

#             key_from = (chunk_id, old_from)
#             key_to = (chunk_id, old_to)

#             if key_from not in node_id_map or key_to not in node_id_map:
#                 continue

#             final_edges.append({
#                 "edge_id": f"edge{global_edge_idx}",
#                 "from": node_id_map[key_from],
#                 "to": node_id_map[key_to]
#             })
#             global_edge_idx += 1
#     else:
#         print("⚠ No edges directory found. Edges = []")

#     # ------------------------------------------------
#     # Positions.chunks
#     # ------------------------------------------------
#     positions_chunks = []
#     for cid, bbox in chunk_bbox_map.items():
#         positions_chunks.append({
#             "chunk_id": cid,
#             "bbox": bbox
#         })

#     # ------------------------------------------------
#     # Output
#     # ------------------------------------------------
#     final = {
#         "canvas": canvas["canvas"],
#         "layout_flow": layout["layout_flow"],
#         "chunks": final_chunks,
#         "nodes": final_nodes,
#         "edges": final_edges,
#         "positions": {
#             "chunks": positions_chunks,
#             "nodes": positions_nodes
#         }
#     }

#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(final, f, ensure_ascii=False, indent=2)

#     print(f"🎉 Final JSON saved to {output_json}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out_dir", required=True)
#     parser.add_argument("--output", required=True)
#     args = parser.parse_args()

#     main(args.out_dir, args.output)


import json
import argparse
from pathlib import Path

# ============================================================
# Utils
# ============================================================

def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def load_edges_file(path: Path):
    """兼容 JSON / JSONL 两种 edge 输出格式"""
    if not path.exists():
        return []

    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # Case 1: 整体 JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "edges" in obj:
            return obj["edges"]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Case 2: JSONL
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

# ============================================================
# Main
# ============================================================

def main(out_dir, output_json, nodes_filename, edges_filename):
    out_dir = Path(out_dir)
    print(f"🔧 Step13 Assembling: Nodes='{nodes_filename}', Edges='{edges_filename}'")

    # ------------------------------------------------
    # 1. Load canvas
    # ------------------------------------------------
    canvas_path = out_dir / "canvas.jsonl"
    if not canvas_path.exists():
        raise FileNotFoundError(f"Missing {canvas_path}")

    # canvas.jsonl 第一行通常是 {"canvas": {...}}
    canvas_data = load_jsonl(canvas_path)
    if not canvas_data:
        raise ValueError("Canvas file is empty")
    canvas = canvas_data[0].get("canvas", {})

    # ------------------------------------------------
    # 2. Load chunk bboxes (from SAM result)
    # ------------------------------------------------
    chunks_bbox_path = out_dir / "sam_results" / "chunks.jsonl"
    if not chunks_bbox_path.exists():
        raise FileNotFoundError(f"Missing {chunks_bbox_path}")

    chunk_items = load_jsonl(chunks_bbox_path)

    # chunk_id → bbox
    chunk_bbox_map = {}
    final_chunks = []

    for c in chunk_items:
        cid = c["chunk_id"]
        bbox = c["bbox"]
        chunk_bbox_map[cid] = {
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2],
            "h": bbox[3],
        }
        final_chunks.append({
            "chunk_id": cid,
            "node_ids": []
        })

    # ------------------------------------------------
    # 3. Load nodes (遍历每个 chunk 文件夹，找指定的文件名)
    # ------------------------------------------------
    nodes_work_dir = out_dir / "nodes_work"
    if not nodes_work_dir.exists():
        raise FileNotFoundError(f"Missing {nodes_work_dir}")

    node_id_map = {}             # (chunk_id, old_node_id) -> new_node_id
    final_nodes = []
    positions_nodes = []
    chunk_to_node_ids = {c["chunk_id"]: [] for c in final_chunks}

    global_node_idx = 0

    # 排序保证 ID 生成顺序稳定
    for chunk_dir in sorted(nodes_work_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        chunk_id = chunk_dir.name
        
        # 【核心修改】使用传入的文件名，而不是写死 nodes_final_local.jsonl
        local_nodes_path = chunk_dir / nodes_filename
        
        if not local_nodes_path.exists():
            # 允许某个 chunk 缺失该 track 的文件（比如 step5 在某些 chunk 没结果）
            continue

        local_nodes = load_jsonl(local_nodes_path)
        if chunk_id not in chunk_bbox_map:
            continue

        chunk_bbox = chunk_bbox_map[chunk_id]

        for n in local_nodes:
            # 兼容：有些中间文件可能没有 node_id，生成一个临时 key
            old_id = n.get("node_id", f"temp_{len(final_nodes)}")
            
            new_id = f"node{global_node_idx}"
            global_node_idx += 1

            node_id_map[(chunk_id, old_id)] = new_id
            chunk_to_node_ids[chunk_id].append(new_id)

            # local bbox → global bbox
            lx, ly, w, h = n.get("bbox", [0, 0, 0, 0])
            global_bbox = {
                "x": chunk_bbox["x"] + lx,
                "y": chunk_bbox["y"] + ly,
                "w": w,
                "h": h,
            }

            positions_nodes.append({
                "node_id": new_id,
                "bbox": global_bbox
            })

            final_nodes.append({
                "node_id": new_id,
                "chunk_id": chunk_id,
                "node_type": n.get("node_type", ""),
                "desc": n.get("desc", ""),
                "render_method": n.get("render_method", ""),
                "vlm_prompt": n.get("vlm_prompt", "")
            })

    # fill chunk.node_ids
    for c in final_chunks:
        c["node_ids"] = chunk_to_node_ids.get(c["chunk_id"], [])

    # ------------------------------------------------
    # 4. Load edges (遍历每个 chunk，找指定的文件名)
    # ------------------------------------------------
    final_edges = []
    global_edge_idx = 0

    for chunk_dir in sorted(nodes_work_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        chunk_id = chunk_dir.name
        
        # 【核心修改】使用传入的文件名
        edge_file = chunk_dir / edges_filename
        
        if not edge_file.exists():
            continue

        edges = load_edges_file(edge_file)

        for e in edges:
            old_from = e.get("from")
            old_to = e.get("to")

            key_from = (chunk_id, old_from)
            key_to = (chunk_id, old_to)

            # 只有当起点和终点都在 node_id_map 里（即 Step 3 加载到了这两个点）才保留边
            if key_from in node_id_map and key_to in node_id_map:
                final_edges.append({
                    "edge_id": f"edge{global_edge_idx}",
                    "from": node_id_map[key_from],
                    "to": node_id_map[key_to]
                })
                global_edge_idx += 1

    # ------------------------------------------------
    # 5. Output
    # ------------------------------------------------
    positions_chunks = []
    for cid, bbox in chunk_bbox_map.items():
        positions_chunks.append({
            "chunk_id": cid,
            "bbox": bbox
        })

    final = {
        "canvas": canvas,
        "chunks": final_chunks,
        "nodes": final_nodes,
        "edges": final_edges,
        "positions": {
            "chunks": positions_chunks,
            "nodes": positions_nodes
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"🎉 Final JSON saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output", required=True)
    
    # 新增参数，带默认值以兼容旧调用
    parser.add_argument("--nodes_file", default="nodes_final_local.jsonl", help="Filename of nodes inside chunk dir")
    parser.add_argument("--edges_file", default="nodes_final_local_edges.jsonl", help="Filename of edges inside chunk dir")
    
    args = parser.parse_args()

    main(args.out_dir, args.output, args.nodes_file, args.edges_file)