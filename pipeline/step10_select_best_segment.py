import os
import subprocess
import json
import shutil
from glob import glob
from pathlib import Path

# ================= 配置区域 =================
# 脚本路径 (假设 V1.5 脚本在同一目录下，命名为 segment_v1_5.py，请根据实际情况修改)
BASE_DIR = Path(__file__).resolve().parent
SAM_V1   = BASE_DIR / "segment_v1.py"
SAM_V1_5 = BASE_DIR / "segment_v1.5.py"
SAM_V2   = BASE_DIR / "segment_v2.py"

SAM_CHECKPOINT = BASE_DIR / "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
# ===========================================


# ------------------------------------------------------------
# 运行 SAM 各个版本
# ------------------------------------------------------------

def run_sam_generic(script_path, img_path, out_dir, label):
    cmd = [
        "python",
        str(script_path),
        "--img", str(img_path),
        "--out", str(out_dir),
        "--ckpt", str(SAM_CHECKPOINT),
        "--model", SAM_MODEL_TYPE,
    ]

    print(f"▶️ Running SAM_{label}: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)

def run_sam_v1(img_path, out_dir):
    run_sam_generic(SAM_V1, img_path, out_dir, "V1")

def run_sam_v1_5(img_path, out_dir):
    run_sam_generic(SAM_V1_5, img_path, out_dir, "V1.5")

def run_sam_v2(img_path, out_dir):
    run_sam_generic(SAM_V2, img_path, out_dir, "V2")


# ------------------------------------------------------------
# 基础工具函数
# ------------------------------------------------------------
def load_jsonl(path):
    nodes = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    nodes.append(json.loads(line))
    return nodes

def bbox_area(b):
    return b[2] * b[3]

def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter <= 0:
        return 0

    area1 = bbox_area(b1)
    area2 = bbox_area(b2)
    return inter / (area1 + area2 - inter)


# ------------------------------------------------------------
# 核心分析逻辑
# ------------------------------------------------------------

def remove_common_intersection(list_A, list_B, list_C):
    """
    找出 A, B, C 三者共同拥有的 bbox (交集)，并从三者中剔除这些共同部分。
    只比较有差异的部分。
    """
    setA = {tuple(n["bbox"]) for n in list_A}
    setB = {tuple(n["bbox"]) for n in list_B}
    setC = {tuple(n["bbox"]) for n in list_C}

    # 计算三者的交集
    common = setA & setB & setC

    # 剔除交集
    new_A = [n for n in list_A if tuple(n["bbox"]) not in common]
    new_B = [n for n in list_B if tuple(n["bbox"]) not in common]
    new_C = [n for n in list_C if tuple(n["bbox"]) not in common]

    return new_A, new_B, new_C


def analyze(nodes):
    if not nodes:
        return {
            "empty": True,
            "overlap_total": 0,
            "max_overlap_area": 0,
            "max_bbox_area": 0
        }

    N = len(nodes)
    overlap_total = 0
    overlap_pair_max_areas = []
    
    areas = [bbox_area(n["bbox"]) for n in nodes]
    max_bbox_area = max(areas) if areas else 0

    # compute overlaps
    for i in range(N):
        bi = nodes[i]["bbox"]
        ai = bbox_area(bi)
        for j in range(i + 1, N):  # 每对只算一次
            bj = nodes[j]["bbox"]
            aj = bbox_area(bj)

            if iou(bi, bj) > 0:
                overlap_total += 1
                overlap_pair_max_areas.append(max(ai, aj))

    if overlap_total > 0:
        max_overlap_area = max(overlap_pair_max_areas)
    else:
        max_overlap_area = 0

    return {
        "empty": False,
        "overlap_total": overlap_total,
        "max_overlap_area": max_overlap_area,
        "max_bbox_area": max_bbox_area,
    }


def decide_best_of_three(candidates):
    """
    candidates: list of dict 
    [
      {'name': 'A', 'stats': statsA, 'root': outA}, 
      {'name': 'B', 'stats': statsB, 'root': outB},
      ...
    ]
    """
    # 1. 过滤掉空集 (如果没有节点，通常不选，除非全是空)
    valid_candidates = [c for c in candidates if not c['stats']['empty']]
    
    # 如果全是空集，随便选一个（比如 A）
    if not valid_candidates:
        return candidates[0]['name'], "All empty, picking default", candidates[0]['root']

    # 2. 定义排序 Key
    def sort_key(c):
        s = c['stats']
        overlap_count = s['overlap_total']
        
        # 规则 1: 重叠数量越少越好 (Primary Key)
        # Python sort 是升序，所以 overlap_count 直接用
        
        # 规则 2 & 3 (Secondary Key)
        if overlap_count == 0:
            # 如果没有重叠：比较最大 bbox 面积，越小越好 (升序)
            # 这满足需求："另两个没重叠，要另两个中最大bbox小的"
            metric_2 = s['max_bbox_area']
        else:
            # 如果有重叠：比较重叠面积，越大越好 (降序)
            # 原代码逻辑是 overlap 相同且 >0 时，Area 大的优先。为了升序排序，取负值。
            metric_2 = -s['max_overlap_area']
            
        return (overlap_count, metric_2)

    # 3. 排序
    # 排序优先级：
    #   第一关键字：overlap_total (升序) -> 0 优先于 1 优先于 5
    #   第二关键字：
    #       - 对于 overlap=0 的组：max_bbox_area (升序) -> 小框优先
    #       - 对于 overlap>0 的组：max_overlap_area (降序) -> 大重叠优先
    valid_candidates.sort(key=sort_key)

    best = valid_candidates[0]
    
    # 生成解释文本
    winner_name = best['name']
    s = best['stats']
    reason = f"Winner overlap={s['overlap_total']}"
    if s['overlap_total'] == 0:
        reason += f", max_bbox={s['max_bbox_area']:.1f} (minimized)"
    else:
        reason += f", max_overlap_area={s['max_overlap_area']:.1f} (maximized)"

    return winner_name, reason, best['root']


# ------------------------------------------------------------
# 批量处理
# ------------------------------------------------------------
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    images = sorted(
        [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    print(f"📁 共检测到 {len(images)} 张图片\n")

    results = {}

    for img_name in images:
        print("=" * 30)
        print(f"🔍 处理图片：{img_name}")

        img_path = os.path.join(input_folder, img_name)

        # 定义三个输出目录
        outA = os.path.join(output_folder, f"{img_name}_samA")   # V1
        outB = os.path.join(output_folder, f"{img_name}_samB")   # V1.5
        outC = os.path.join(output_folder, f"{img_name}_samC")   # V2

        # 运行三个版本的 SAM
        # (如果在已有结果上重跑，可以加 os.path.exists 判断跳过运行)
        run_sam_v1(img_path, outA)
        run_sam_v1_5(img_path, outB)
        run_sam_v2(img_path, outC)

        # 读取结果
        pathA = os.path.join(outA, "nodes.jsonl")
        pathB = os.path.join(outB, "nodes.jsonl")
        pathC = os.path.join(outC, "nodes.jsonl")

        nodesA = load_jsonl(pathA)
        nodesB = load_jsonl(pathB)
        nodesC = load_jsonl(pathC)

        # 移除三者共同的部分，只分析差异
        uniqA, uniqB, uniqC = remove_common_intersection(nodesA, nodesB, nodesC)

        # 分析指标
        statsA = analyze(uniqA)
        statsB = analyze(uniqB)
        statsC = analyze(uniqC)

        # 构造候选列表
        candidates = [
            {'name': 'A (V1)',   'stats': statsA, 'root': outA},
            {'name': 'B (V1.5)', 'stats': statsB, 'root': outB},
            {'name': 'C (V2)',   'stats': statsC, 'root': outC}
        ]

        # 决策
        better_name, reason, best_root = decide_best_of_three(candidates)

        # 复制最佳结果
        result_folder = Path(output_folder) / f"{img_name}_result"
        # 使用 dirs_exist_ok=True 覆盖旧结果
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        shutil.copytree(best_root, result_folder)

        # 修正 chunk_id
        chunk_id = img_name.rsplit(".", 1)[0]
        final_nodes_path = os.path.join(result_folder, "nodes.jsonl")
        
        if os.path.exists(final_nodes_path):
            tmp_lines = []
            with open(final_nodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        d["chunk_id"] = chunk_id
                        tmp_lines.append(json.dumps(d, ensure_ascii=False))
            with open(final_nodes_path, "w", encoding="utf-8") as f:
                f.write("\n".join(tmp_lines))

        # 记录日志
        results[img_name] = {
            "winner": better_name,
            "reason": reason,
            "stats": {
                "A_rem": len(uniqA), "A_overlap": statsA['overlap_total'],
                "B_rem": len(uniqB), "B_overlap": statsB['overlap_total'],
                "C_rem": len(uniqC), "C_overlap": statsC['overlap_total'],
            }
        }

        print(f"✨ 最佳版本：{better_name}")
        print(f"📌 原因：{reason}")
        print("-" * 10)

    # 保存最终统计 JSON
    out_json = os.path.join(output_folder, "best_sam_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 全部完成！统计结果已保存：{out_json}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入图片文件夹路径")
    parser.add_argument("--output", required=True, help="输出文件夹路径")
    args = parser.parse_args()

    process_folder(args.input, args.output)

