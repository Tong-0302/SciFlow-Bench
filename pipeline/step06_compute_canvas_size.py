import json
import argparse

def compute_canvas_size(chunks):
    """
    根据所有 chunk 的 bbox 推算画布尺寸
    bbox: [x, y, w, h]
    返回 (width, height)
    """
    max_right = 0
    max_bottom = 0

    for c in chunks:
        bbox = c["bbox"]
        x, y, w, h = bbox
        right = x + w
        bottom = y + h

        if right > max_right:
            max_right = right
        if bottom > max_bottom:
            max_bottom = bottom

    return max_right, max_bottom


def load_jsonl(path):
    """读取 jsonl 文件"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def save_canvas_jsonl(path, width, height):
    """保存画布信息到 jsonl"""
    canvas_info = {
        "canvas": {
            "width": width,
            "height": height,
            "unit": "px"
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(canvas_info, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 chunk.jsonl")
    parser.add_argument("--output", required=True, help="输出 canvas.jsonl")
    args = parser.parse_args()

    chunks = load_jsonl(args.input)
    width, height = compute_canvas_size(chunks)
    save_canvas_jsonl(args.output, width, height)

    print(f"Canvas size computed: width={width}, height={height} (px)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
