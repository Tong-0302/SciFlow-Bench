import os
import subprocess
import torch
import numpy as np
import cv2
import argparse
import json
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 显示 SAM mask
# ---------------------------------------------------------
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        img = np.ones((m.shape[0], m.shape[1], 3))
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.5)))


# ---------------------------------------------------------
# bbox-based NMS：保留小框
# ---------------------------------------------------------
def nms_keep_small(masks, iou_thresh=0.1, contain_thresh=0.9):

    def compute_inter_iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        iou = inter / union if union > 0 else 0
        return inter, area1, area2, iou

    def contain_ratio(inner, outer):
        inter, inner_area, _, _ = compute_inter_iou(inner, outer)
        return inter / inner_area if inner_area > 0 else 0

    filtered = []

    for m1 in masks:
        b1 = m1["bbox"]
        area1 = b1[2] * b1[3]

        keep = True
        to_remove = []

        for j, m2 in enumerate(filtered):
            b2 = m2["bbox"]
            area2 = b2[2] * b2[3]

            inter, _, _, iou = compute_inter_iou(b1, b2)
            c1 = contain_ratio(b1, b2)
            c2 = contain_ratio(b2, b1)

            # 完全包裹 → 保留小框
            if c1 > contain_thresh or c2 > contain_thresh:
                if area1 < area2:
                    to_remove.append(j)
                else:
                    keep = False
                break

            # 部分重叠 → 保留小框
            if iou > iou_thresh:
                if area1 < area2:
                    to_remove.append(j)
                else:
                    keep = False
                break

        # 删除被新的小框取代的旧框
        for j in reversed(to_remove):
            filtered.pop(j)

        if keep:
            filtered.append(m1)

    return filtered


# ---------------------------------------------------------
# 最大连通域
# ---------------------------------------------------------
def extract_largest_cc(mask):
    mask_uint8 = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8)

    if num_labels <= 1:
        return mask_uint8

    max_area = 0
    max_label = 1
    for lbl in range(1, num_labels):
        area = np.sum(labels == lbl)
        if area > max_area:
            max_area = area
            max_label = lbl

    return (labels == max_label).astype(np.uint8)
def crop_by_cc(image, mask):
    mask_cc = extract_largest_cc(mask)

    ys, xs = np.where(mask_cc == 1)
    if len(xs) == 0:
        return None, None

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()

    # 1. 还是裁剪原图（这一步没变）
    cropped_rgb = image[y1:y2+1, x1:x2+1]
    
    # 获取裁剪后的尺寸
    Hc, Wc = cropped_rgb.shape[:2]

    # 2. 【修改点】创建全不透明的底图
    rgba = np.zeros((Hc, Wc, 4), dtype=np.uint8)
    
    # 3. 【核心修改】直接填入原图，不乘 mask！
    rgba[..., :3] = cropped_rgb 
    
    # 4. 【核心修改】Alpha 通道全部设为 255 (完全不透明)
    rgba[..., 3] = 255  

    bbox = (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))
    return rgba, bbox

# ---------------------------------------------------------
# 主函数（chunk 版本）
# ---------------------------------------------------------
def segment_and_save_chunks(image_path, sam_checkpoint, output_dir, model_type='vit_h'):

    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # 读图
    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    # SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.88,
        min_mask_region_area=50
    )

    masks = mask_generator.generate(image)

    # 过滤大块（chunk = 大区域）
    area_limit = 0.02 * H * W
    masks = [m for m in masks if m["area"] > area_limit]

    print(f"过滤后剩 {len(masks)} 个大 mask")

    # NMS
    masks = nms_keep_small(masks)
    print(f"NMS 后剩 {len(masks)}")

    chunks = []
    red_boxes = []

    # 保存每个 chunk
    for i, m in enumerate(masks):
        mask = m["segmentation"]

        # mask 贴边
        rgba, tight_bbox = crop_by_cc(image, mask)
        if rgba is None:
            continue

        x, y, w, h = tight_bbox
        print(f"chunk{i}: shape={rgba.shape}, tight_bbox={tight_bbox}")

        save_path = os.path.join(chunks_dir, f"chunk{i}.png")
        cv2.imwrite(save_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        chunks.append({
            "chunk_id": f"chunk{i}",
            "bbox": [x, y, w, h]
        })
        red_boxes.append(tight_bbox)

    # -----------------------------
    # ⚠️ 面积检查：如果所有 chunk 面积 < 原图 3/4，则放弃分块
    # -----------------------------
    total_chunk_area = sum(w * h for (_, _, w, h) in red_boxes)
    full_area = H * W

    if total_chunk_area < 0.75 * full_area:
        print("⚠️ 所有 chunk 面积之和 < 75% 的原图面积 → 取消分块，改为整图单块模式")

        # 清空 chunks 目录
        for fname in os.listdir(chunks_dir):
            os.remove(os.path.join(chunks_dir, fname))

        # 直接将整图当成 chunk0
        full_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        full_rgba[..., :3] = image
        full_rgba[..., 3] = 255  # 完全不透明

        full_path = os.path.join(chunks_dir, "chunk0.png")
        cv2.imwrite(full_path, cv2.cvtColor(full_rgba, cv2.COLOR_RGBA2BGRA))

        chunks = [{
            "chunk_id": "chunk0",
            "bbox": [0, 0, W, H]
        }]
        red_boxes = [(0, 0, W, H)]  # 给可视化画框用

    # 写 JSONL
    jsonl_path = os.path.join(output_dir, "chunks.jsonl")
    with open(jsonl_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"📄 共保存 {len(chunks)} 个 chunk → {jsonl_path}")

    # 可视化图
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    show_anns(masks)

    ax = plt.gca()
    for (x, y, w, h) in red_boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, edgecolor="red", facecolor="none", linewidth=2)
        )

    plt.axis("off")
    seg_path = os.path.join(output_dir, "segmentation_result.png")
    plt.savefig(seg_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    print(f"🖼 segmentation_result.png 已保存到：{seg_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="vit_h")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"✅ GPU 状态: 正常 | 设备名: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU 状态: 未检测到 (正在使用 CPU，速度会极慢)")

    segment_and_save_chunks(args.img, args.ckpt, args.out, args.model)
