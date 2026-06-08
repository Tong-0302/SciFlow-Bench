import json
import argparse
import re


def normalize(text):
    """删除空格、标点，仅保留字母数字并转小写"""
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def enforce_order(node):
    """统一字段顺序 & 填补缺失字段"""
    return {
        "node_id": node.get("node_id", ""),
        "chunk_id": node.get("chunk_id", ""),
        "bbox": node.get("bbox", []),
        "desc": node.get("desc", ""),
        "vlm_prompt": node.get("vlm_prompt", ""),
        "node_type": node.get("node_type", ""),     # OCR节点通常没有 -> 填 ""
        "render_method": node.get("render_method", "")
    }


def clean_and_merge_jsonl(jsonl1, jsonl2, out_path):

    # ----- STEP A: 构建 jsonl1 中可删除的规范化 desc 集 -----
    jsonl1_norm_desc = {
        normalize(n["desc"]) for n in jsonl1 if n.get("desc", "") != ""
    }

    # ----- STEP B: jsonl2 的规范化 desc 列表 -----
    jsonl2_norm_list = [normalize(n.get("desc", "")) for n in jsonl2]

    # ----- STEP C: 多段拼接匹配 -----
    delete_indices = set()

    for start in range(len(jsonl2)):
        concat = ""
        for end in range(start, len(jsonl2)):
            concat += jsonl2_norm_list[end]
            if concat in jsonl1_norm_desc:
                for idx in range(start, end + 1):
                    delete_indices.add(idx)
                break

    # ----- STEP D: jsonl2 过滤 -----

    # 用于大小写敏感 substring 匹配
    jsonl1_raw_desc_list = [n.get("desc", "") for n in jsonl1]

    jsonl2_filtered = []
    for i, node2 in enumerate(jsonl2):

        desc_raw = node2.get("desc", "")
        desc_norm = normalize(desc_raw)

        # A. 属于多段拼接删除的索引
        if i in delete_indices:
            continue

        # B. 空 desc 直接跳过
        if desc_raw == "":
            continue

        # C. 完全相同（normalize 后相同） → 删除
        if desc_norm in jsonl1_norm_desc:
            continue

        # D. ⭐ NEW RULE (大小写敏感):
        # jsonl2 的 desc_raw 是 jsonl1 某个 desc_raw 的子串 → 删除
        to_delete = False
        for d1_raw in jsonl1_raw_desc_list:
            if desc_raw != "" and desc_raw in d1_raw:
                to_delete = True
                break

        if to_delete:
            continue

        # E. 保留 node2
        node2["node_type"] = "text_block"
        jsonl2_filtered.append(node2)


    # ----- STEP E: 合并 -----
    merged_nodes = []

    # 保证 chunk_id 不丢失
    for n in jsonl1:
        n["render_method"] = "vlm"
        merged_nodes.append(enforce_order(n))

    for n in jsonl2_filtered:
        n["render_method"] = "pptx"
        merged_nodes.append(enforce_order(n))

    # ----- STEP F: 重新编号 node_id -----
    final_nodes = []
    for i, node in enumerate(merged_nodes):
        node["node_id"] = f"node{i}"
        final_nodes.append(node)

    # ----- STEP G: 写入 JSONL -----
    with open(out_path, 'w', encoding='utf-8') as f:
        for n in final_nodes:
            f.write(json.dumps(enforce_order(n), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', required=True)
    parser.add_argument('--input2', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.input1, 'r', encoding='utf-8') as f1:
        jsonl1 = [json.loads(line.strip()) for line in f1.readlines()]

    with open(args.input2, 'r', encoding='utf-8') as f2:
        jsonl2 = [json.loads(line.strip()) for line in f2.readlines()]

    clean_and_merge_jsonl(jsonl1, jsonl2, args.output)


if __name__ == "__main__":
    main()
