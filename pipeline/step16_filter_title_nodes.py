import json
import argparse
import re

def strip_leading_bracket(s):
    """
    去掉字符串开头的括号及里面的内容，如：
    "(c) Title" -> "Title"
    "(ab123) Something" -> "Something"
    "[12] Something" -> "Something"

    只移除第一个括号对，且必须在开头。
    """
    if s is None:
        return ""
    s = s.strip()
    # 匹配字符串开头的 (xxx) 或 [xxx] 或 {xxx}
    return re.sub(r'^[\(\[\{][^\)\]\}]*[\)\]\}]\s*', '', s)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def normalize(s: str) -> str:
    if s is None:
        return ""
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True, help="当前 chunk 的 nodes.jsonl")
    parser.add_argument("--titles", required=True, help="chunks_with_titles.jsonl")
    parser.add_argument("--out", required=True, help="输出路径")
    args = parser.parse_args()

    # load
    nodes = load_jsonl(args.nodes)
    titles = load_jsonl(args.titles)

    # 当前 nodes.jsonl 只有一个 chunk_id
    if len(nodes) == 0:
        print("nodes.jsonl 是空的")
        return

    current_chunk = nodes[0]["chunk_id"]
    print(f"处理 chunk_id = {current_chunk}")

    # 找到这个 chunk 的 title
    raw_title = ""
    for t in titles:
        if t["chunk_id"] == current_chunk:
            raw_title = t.get("titles", "")
            break

    # title 分割成多个子 title
    parts = [p.strip() for p in raw_title.split(";") if p.strip()]
    normalized_titles = [normalize(strip_leading_bracket(p)) for p in parts]

    # 为了连续匹配，给 nodes 做 normalize
    desc_norm_list = [normalize(strip_leading_bracket(n["desc"])) for n in nodes]
    n = len(nodes)

    to_skip = set()  # 全局 index

    # 对每个子 title 做连续匹配
    for title_norm in normalized_titles:
        i = 0
        while i < n:
            concat = ""
            j = i
            matched = False
            while j < n:
                concat += desc_norm_list[j]

                if concat == title_norm:
                    # 删掉 i~j
                    for k in range(i, j + 1):
                        to_skip.add(k)
                    matched = True
                    break

                if not title_norm.startswith(concat):
                    break

                j += 1

            if matched:
                i = j + 1
            else:
                i += 1

    # 输出
    kept = []
    for i, node in enumerate(nodes):
        if i not in to_skip:
            kept.append(node)

    with open(args.out, "w", encoding="utf-8") as f:
        for item in kept:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"原有节点 {len(nodes)}, 删除 {len(to_skip)}, 保留 {len(kept)}")
    print(f"输出到 {args.out}")

if __name__ == "__main__":
    main()
