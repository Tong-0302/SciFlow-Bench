import os
import json
import argparse
import re
from tqdm import tqdm


def load_chunk_titles(chunks_jsonl):
    """
    读取 chunks_with_titles_and_summary.jsonl：
    - 优先使用 "titles": [..] 这个数组
    - 如果只有 "title": "A; B" 这种，就按 ; 分割
    """
    chunk_titles = {}
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj["chunk_id"]

            if "titles" in obj and isinstance(obj["titles"], list):
                titles = [t.strip() for t in obj["titles"] if t and t.strip()]
            else:
                raw = obj.get("titles", "") or ""
                titles = [t.strip() for t in raw.split(";") if t.strip()]

            chunk_titles[cid] = titles
    return chunk_titles


def is_title_by_whole_word(desc, titles):
    """
    仅当 desc 作为“完整词/完整短语”出现在标题中时，才认为是标题节点。

    例：
      title = "Feature Fusion (FF)"
      desc  = "Feature"           -> True
      desc  = "Fusion"            -> True
      desc  = "Feature Fusion"    -> True
      desc  = "FF"                -> True
      desc  = "Fea"               -> False
    """
    if not desc:
        return False

    desc = desc.strip()
    if not desc:
        return False

    pattern = r"\b" + re.escape(desc) + r"\b"

    for t in titles:
        if re.search(pattern, t):
            return True
    return False


def remove_title_nodes(chunks_jsonl, nodes_jsonl, out_jsonl):
    chunk_titles = load_chunk_titles(chunks_jsonl)

    nodes = []
    with open(nodes_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(json.loads(line))

    final_nodes = []
    removed = 0

    for node in tqdm(nodes, desc="规则删除标题节点"):
        cid = node["chunk_id"]
        desc = (node.get("desc") or "").strip()

        if not desc:
            # 没有 desc 的节点不可能是标题，直接保留
            final_nodes.append(node)
            continue

        titles = chunk_titles.get(cid, [])

        if titles and is_title_by_whole_word(desc, titles):
            removed += 1
            print(f"🗑 删除标题节点 {node['node_id']}  desc='{desc}'  chunk_id={cid}")
            continue

        final_nodes.append(node)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for n in final_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    print("==================================")
    print("删除标题节点数:", removed)
    print("剩余节点数:", len(final_nodes))
    print("输出文件:", out_jsonl)
    print("==================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True,
                        help="chunks_with_titles_and_summary.jsonl")
    parser.add_argument("--nodes", required=True,
                        help="nodes.jsonl（含 desc 和 chunk_id）")
    parser.add_argument("--out", required=True,
                        help="输出 nodes 去标题后的 jsonl")
    args = parser.parse_args()

    remove_title_nodes(args.chunks, args.nodes, args.out)
