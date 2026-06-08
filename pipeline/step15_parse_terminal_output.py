import subprocess
import json
import re
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="运行 OCR 脚本并解析终端输出为 JSONL 格式")
    parser.add_argument("--input", required=True, help="输入的 Python 脚本路径（例如 run_dpsk_ocr.py）")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--img", default=None, help="可选：传给 OCR 脚本的 --img 路径")
    return parser.parse_args()

def run_script(script_path, img_path=None):
    # 如果提供了 img，就以 `python script --img xxx` 的方式运行
    if img_path:
        cmd = ["python", script_path, "--img", img_path]
    else:
        cmd = ["python", script_path]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout

def parse_terminal_output(content):
    # 只匹配严格格式：<|ref|>xxx<|/ref|><|det|>[[x1, x2, x3, x4]]<|/det|>
    pattern = (
        r"<\|ref\|>(.*?)<\|/ref\|>"
        r"<\|det\|>"
        r"\[\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*\]"
        r"<\|/det\|>"
    )

    matches = re.findall(pattern, content, flags=re.S)

    jsonl_data = []
    for desc, x1, x2, x3, x4 in matches:
        bbox = [float(x1), float(x2), float(x3), float(x4)]
        jsonl_data.append({
            "desc": desc.strip(),
            "bbox": bbox
        })

    return jsonl_data


def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def main():
    args = parse_args()

    output_content = run_script(args.input, args.img)
    jsonl_data = parse_terminal_output(output_content)

    chunk_id = None
    if args.img:
        chunk_id = os.path.splitext(os.path.basename(args.img))[0]  # chunk0.png → chunk0
    if chunk_id:
        for item in jsonl_data:
            item["chunk_id"] = chunk_id

    save_to_jsonl(jsonl_data, args.output)

    print(f"数据已成功保存到 {args.output}，共 {len(jsonl_data)} 行")

if __name__ == "__main__":
    main()

