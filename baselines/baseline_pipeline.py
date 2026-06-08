import os
import json
import subprocess
import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv() 

BASE = Path(__file__).parent.resolve()


def run(cmd, env=None):
    print("▶️", " ".join(map(str, cmd)))
    if env is None:
        env = os.environ.copy()
    env["DF_API_KEY"] = os.getenv("DF_API_KEY", "")
    env["DF_API_URL"] = os.getenv("DF_API_URL", "http://localhost:3888/v1")
    env.setdefault("DF_MODEL", os.getenv("DF_MODEL", "gpt-4o"))

    subprocess.run(cmd, check=True, env=env)


def file_not_empty(p: Path):
    return p.exists() and p.stat().st_size > 10


def jsonl_not_empty(p: Path):
    if not p.exists():
        return False
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


# ================================================================
# cleanup
# ================================================================
def cleanup_workspace(out_dir: Path):
    print(f"\n🧹 开始整理最终结果到: {out_dir}")

    final_files = {
        "canvas_full.json",
        "segmentation_result.png",
        "intermediates",
    }

    # --- sam segmentation result ---
    src_seg = out_dir / "sam_results" / "segmentation_result.png"
    dst_seg = out_dir / "segmentation_result.png"
    if src_seg.exists():
        dst_seg.unlink(missing_ok=True)
        shutil.move(str(src_seg), str(dst_seg))
        final_files.add("segmentation_result.png")
        print("   ✅ segmentation_result.png")

    # --- prepare intermediates dir ---
    inter_dir = out_dir / "intermediates"
    inter_dir.mkdir(exist_ok=True)
    final_files.add("intermediates")

    # --- copy SAM sub results ---
    nodes_sam_root = out_dir / "nodes_sam"
    if nodes_sam_root.exists():
        for chunk_dir in nodes_sam_root.glob("*_result"):
            seg = chunk_dir / "segmentation_result.png"
            if seg.exists():
                dst = inter_dir / "nodes_sam" / chunk_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(seg, dst / "segmentation_result.png")

    # --- copy OCR ---
    nodes_work = out_dir / "nodes_work"
    if nodes_work.exists():
        for chunk_dir in nodes_work.iterdir():
            ocr8 = chunk_dir / "ocr_step8.jsonl"
            if ocr8.exists():
                dst = inter_dir / "nodes_work" / chunk_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ocr8, dst / "ocr_step8.jsonl")

    print("   ✅ intermediates done")

    # --- clear others ---
    for item in out_dir.iterdir():
        if item.name in final_files:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)

    print("✨ 清理完成！")


# ================================================================
# 主流程
# ================================================================
def main(image_path: str, out_dir: str):
    image_path = Path(image_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 AutoBench IMAGE MODE")
    print(f"Input Image: {image_path}")
    print(f"Output Dir: {out_dir}")

    # ============================================================
    # STEP 1.5 — Layout Detection [已跳过]
    # ============================================================
    layout_path = out_dir / "layout.jsonl"
    print("【配置跳过】STEP1.5 Layout Detection")
    # if not jsonl_not_empty(layout_path):
    #     print("\n=== 🚀 STEP1.5: Layout Detection ===")
    #     run([
    #         "python",
    #         str(BASE / "step1.5_detect_layout.py"),
    #         "--img", str(image_path),
    #         "--out", str(layout_path)
    #     ])
    # else:
    #     print("【跳过】STEP1.5 已完成")

    # ============================================================
    # STEP 2 — SAM Segment
    # ============================================================
    sam_root = out_dir / "sam_results"
    sam_root.mkdir(exist_ok=True)

    chunks_bbox = sam_root / "chunks.jsonl"
    chunks_dir = sam_root / "chunks"

    if not (jsonl_not_empty(chunks_bbox) and chunks_dir.exists()):
        print("\n=== 🚀 STEP2: SAM Segment ===")
        run([
            "python",
            str(BASE / "step2_sam_segment.py"),
            "--img", str(image_path),
            "--out", str(sam_root),
            "--ckpt", str(BASE / "sam_vit_h_4b8939.pth")
        ])
    else:
        print("【跳过】STEP2 已完成")

    # ============================================================
    # STEP 2.1 — Canvas Size
    # ============================================================
    canvas_file = out_dir / "canvas.jsonl"

    if not file_not_empty(canvas_file):
        print("\n=== 🚀 STEP2.1: Canvas Size ===")
        run([
            "python",
            str(BASE / "step2.1_compute_canvas_size.py"),
            "--input", str(chunks_bbox),
            "--output", str(canvas_file)
        ])
    else:
        print("【跳过】STEP2.1 已完成")

    # ============================================================
    # STEP 2.5+2.6 — VLM Title + Summary [已跳过]
    # ============================================================
    # 原始逻辑：chunks_summary = sam_root / "chunks_with_titles_and_summary.jsonl"
    # 修改逻辑：直接复用 chunks.jsonl，后续步骤会将其复制为 chunks_with_titles_and_summary.jsonl
    print("【配置跳过】STEP2.5 + 2.6: Title + Summary")
    
    # 强制让 summary 变量指向 chunks.jsonl，这样下面的复制逻辑会把 chunks.jsonl 复制过去
    chunks_summary = chunks_bbox 

    # if not jsonl_not_empty(chunks_summary):
    #     print("\n=== 🚀 STEP2.5 + 2.6: Title + Summary ===")
    #     run([
    #         "python",
    #         str(BASE / "step2.5_step2.6_chunk_title_and_summary.py"),
    #         "--chunks", str(chunks_bbox),
    #         "--imgs", str(chunks_dir),
    #         "--out", str(chunks_summary)
    #     ])
    # else:
    #     print("【跳过】STEP2.5+2.6 已完成")

    # ============================================================
    # [关键修复] Prepare Chunks Dir for Step 13
    # 确保 step13 能在 out_dir/chunks 找到文件
    # ============================================================
    final_chunks_dir = out_dir / "chunks"
    final_chunks_dir.mkdir(exist_ok=True)
    
    # 1. 复制 JSONL (因为跳过了2.5，这里会把原始 chunks.jsonl 复制成 chunks_with_titles_and_summary.jsonl)
    #    这样 Step 13 依然能读到文件，只是里面没有 title 和 summary 字段，通常不影响结构
    if chunks_summary.exists():
        dst_jsonl = final_chunks_dir / "chunks_with_titles_and_summary.jsonl"
        # 如果目标不存在或大小不同，则复制
        if not dst_jsonl.exists() or dst_jsonl.stat().st_size != chunks_summary.stat().st_size:
            shutil.copy2(chunks_summary, dst_jsonl)
            print(f"📦 Copied chunks jsonl to {final_chunks_dir}")
    
    # 2. 复制图片 (chunk0.png ...)
    if chunks_dir.exists():
        for png in chunks_dir.glob("chunk*.png"):
            dst_png = final_chunks_dir / png.name
            if not dst_png.exists():
                shutil.copy2(png, dst_png)

    # ============================================================
    # STEP 3 — Best SAM Nodes
    # ============================================================
    nodes_sam_root = out_dir / "nodes_sam"
    nodes_sam_root.mkdir(exist_ok=True)

    if not any(nodes_sam_root.glob("*_result")):
        print("\n=== 🚀 STEP3: Best SAM ===")
        run([
            "python",
            str(BASE / "step3_choose_best_sam.py"),
            "--input", str(chunks_dir),
            "--output", str(nodes_sam_root)
        ])
    else:
        print("【跳过】STEP3 已完成")

    # ============================================================
    # STEP 4 ~ 10 per chunk
    # ============================================================
    nodes_work_root = out_dir / "nodes_work"
    nodes_work_root.mkdir(exist_ok=True)

    chunk_pngs = sorted(chunks_dir.glob("chunk*.png"))
    print(f"\n📌 Found {len(chunk_pngs)} chunks")

    for chunk_img in chunk_pngs:
        cid = chunk_img.stem
        work_dir = nodes_work_root / cid
        work_dir.mkdir(exist_ok=True)

        nodes_jsonl = nodes_sam_root / (chunk_img.name + "_result") / "nodes.jsonl"

        print(f"\n=== 🔍 PROCESS CHUNK {cid} ===")

        # STEP4
        step4 = work_dir / "nodes_step4.jsonl"
        if not jsonl_not_empty(step4):
            run([
                "python",
                str(BASE / "step4_describe_node_with_type_delete_meaningless.py"),
                "--nodes", str(nodes_jsonl),
                "--imgs", str(nodes_jsonl.parent / "nodes"),
                "--out", str(step4)
            ])
        else:
            print("  【跳过】STEP4")

        # STEP5
        step5 = work_dir / "nodes_step5.jsonl"
        if not jsonl_not_empty(step5):
            run([
                "python",
                str(BASE / "step5_delete_sam.py"),
                "--nodes", str(step4),
                "--out", str(step5)
            ])
        else:
            print("  【跳过】STEP5")

        # STEP6 [已跳过]
        step6 = work_dir / "nodes_step6.jsonl"
        print("  【配置跳过】STEP6 (VLM Filter)")
        # if not jsonl_not_empty(step6):
        #     run([
        #         "python",
        #         str(BASE / "step6_title_vlm_delete.py"),
        #         "--chunks", str(chunks_summary),
        #         "--nodes", str(step5),
        #         "--out", str(step6)
        #     ])
        # else:
        #     print("  【跳过】STEP6")

        # STEP7 (OCR raw)
        step7 = work_dir / "ocr_raw.jsonl"
        if not jsonl_not_empty(step7):
            run([
                "python",
                str(BASE / "step7_parse_terminal_output.py"),
                "--input", str(BASE / "run_dpsk_ocr.py"),
                "--img", str(chunk_img),
                "--output", str(step7)
            ])
        else:
            print("  【跳过】STEP7")

        # STEP8 [已跳过]
        step8 = work_dir / "ocr_step8.jsonl"
        print("  【配置跳过】STEP8 (OCR Filter)")
        # if not jsonl_not_empty(step8):
        #     run([
        #         "python",
        #         str(BASE / "step8_filter_title_nodes.py"),
        #         "--nodes", str(step7),
        #         "--titles", str(chunks_summary),
        #         "--out", str(step8)
        #     ])
        # else:
        #     print("  【跳过】STEP8")

        # STEP10 (Modified Input)
        step10 = work_dir / "nodes_final_local.jsonl"
        if not jsonl_not_empty(step10):
            # 修改：因为跳过了 step6 和 step8
            # input1: step6 -> step5
            # input2: step8 -> step7
            print("  🚀 STEP10: Merge (Using Step5 + Step7 directly)")
            
            run([
                "python",
                str(BASE / "step10_merge.py"),
                "--input1", str(step5), # Modified
                "--input2", str(step7), # Modified
                "--output", str(step10)
            ])
        else:
            print("  【跳过】STEP10")

    # ============================================================
    # STEP 11 — Mermaid
    # ============================================================
    mermaid_root = out_dir / "mermaid"
    mermaid_root.mkdir(exist_ok=True)

    if not any(mermaid_root.glob("*.mmd")):
        print("\n=== 🚀 STEP11: Mermaid ===")
        env = os.environ.copy()
        env["DF_MODEL"] = "gemini-2.5-flash-image-preview"
        run([
            "python",
            str(BASE / "step11_chunk_mermaid.py"),
            "--imgs", str(chunks_dir),
            "--out", str(mermaid_root)
        ], env=env)
    else:
        print("【跳过】STEP11 已完成")

    # ============================================================
    # STEP 12 — Extract Edges
    # ============================================================
    for chunk_img in chunk_pngs:
        cid = chunk_img.stem
        mmd_file = mermaid_root / f"{cid}.mmd"
        edge_tmp = nodes_work_root / cid / "nodes_final_local_edges.jsonl"

        if not mmd_file.exists():
            print(f"⚠ {cid} 无 mermaid，跳过 Step12")
            continue

        if not jsonl_not_empty(edge_tmp):
            print(f"\n=== 🚀 STEP12: Extract edges for {cid} ===")
            env = os.environ.copy()
            env["DF_MODEL"] = "gemini-2.5-flash-image-preview"
            run([
                "python",
                str(BASE / "step12_extract_edges.py"),
                "--nodes", str(nodes_work_root / cid / "nodes_final_local.jsonl"),
                "--mmd", str(mmd_file)
            ], env=env)
        else:
            print(f"  【跳过】STEP12 {cid}")

    # ============================================================
    # STEP 12.5 — Global Nodes
    # ============================================================
    nodes_dir = out_dir / "nodes"
    nodes_dir.mkdir(exist_ok=True)

    global_nodes_file = nodes_dir / "nodes_final.jsonl"

    if not jsonl_not_empty(global_nodes_file):
        print("\n=== 🚀 STEP12.5: Build Global Nodes ===")
        all_nodes = []
        for chunk_img in chunk_pngs:
            local = nodes_work_root / chunk_img.stem / "nodes_final_local.jsonl"
            if jsonl_not_empty(local):
                with open(local, "r") as f:
                    for line in f:
                        all_nodes.append(json.loads(line))

        id_map = {}
        for i, n in enumerate(all_nodes):
            old = n["node_id"]
            new = f"node{i}"
            id_map[old] = new
            n["node_id"] = new

        with open(global_nodes_file, "w") as f:
            for n in all_nodes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")

        with open(out_dir / "node_id_map.json", "w") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)

    else:
        print("【跳过】STEP12.5 已完成")

    # ============================================================
    # STEP 13 — Assemble Final JSON
    # ============================================================
    canvas_full = out_dir / "canvas_full.json"

    if not file_not_empty(canvas_full):
        print("\n=== 🚀 STEP13: Assemble Final JSON ===")
        run([
            "python",
            str(BASE / "step13_assemble_final_json.py"),
            "--out_dir", str(out_dir),
            "--output", str(canvas_full)
        ])
    else:
        print("【跳过】STEP13 已完成")

    # ============================================================
    # CLEANUP
    # ============================================================
    #cleanup_workspace(out_dir)
    print("\n🎉 ALL DONE.")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="输入图片路径")
    parser.add_argument("--out", required=True, help="输出目录")
    args = parser.parse_args()

    main(args.img, args.out)
    
# python AutoBench/dataflow_agent/bench/baseline_autobench.py --img 123/nlp_2025/2508.06810v1/gemini3.png --out 123/nlp_2025/2508.06810v1/gemini3_autobench
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# import os
# import json
# import subprocess
# import argparse
# import shutil
# from pathlib import Path

# from dotenv import load_dotenv
# load_dotenv() 

# BASE = Path(__file__).parent.resolve()


# def run(cmd, env=None):
#     print("▶️", " ".join(map(str, cmd)))
#     if env is None:
#         env = os.environ.copy()
#     env["DF_API_KEY"] = os.getenv("DF_API_KEY", "")
#     env["DF_API_URL"] = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
#     env.setdefault("DF_MODEL", os.getenv("DF_MODEL", "gpt-4o"))

#     subprocess.run(cmd, check=True, env=env)


# def file_not_empty(p: Path):
#     return p.exists() and p.stat().st_size > 10


# def jsonl_not_empty(p: Path):
#     if not p.exists():
#         return False
#     with open(p, "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 return True
#     return False


# # ================================================================
# # cleanup
# # ================================================================
# # def cleanup_workspace(out_dir: Path):
# #     print(f"\n🧹 开始整理最终结果到: {out_dir}")

# #     final_files = {
# #         "canvas_full.json",
# #         "segmentation_result.png",
# #         "intermediates",
# #     }

# #     # --- sam segmentation result ---
# #     src_seg = out_dir / "sam_results" / "segmentation_result.png"
# #     dst_seg = out_dir / "segmentation_result.png"
# #     if src_seg.exists():
# #         dst_seg.unlink(missing_ok=True)
# #         shutil.move(str(src_seg), str(dst_seg))
# #         final_files.add("segmentation_result.png")
# #         print("   ✅ segmentation_result.png")

# #     # --- prepare intermediates dir ---
# #     inter_dir = out_dir / "intermediates"
# #     inter_dir.mkdir(exist_ok=True)
# #     final_files.add("intermediates")

# #     # --- copy SAM sub results ---
# #     nodes_sam_root = out_dir / "nodes_sam"
# #     if nodes_sam_root.exists():
# #         for chunk_dir in nodes_sam_root.glob("*_result"):
# #             seg = chunk_dir / "segmentation_result.png"
# #             if seg.exists():
# #                 dst = inter_dir / "nodes_sam" / chunk_dir.name
# #                 dst.mkdir(parents=True, exist_ok=True)
# #                 shutil.copy2(seg, dst / "segmentation_result.png")

# #     # --- copy OCR ---
# #     nodes_work = out_dir / "nodes_work"
# #     if nodes_work.exists():
# #         for chunk_dir in nodes_work.iterdir():
# #             ocr8 = chunk_dir / "ocr_step8.jsonl"
# #             if ocr8.exists():
# #                 dst = inter_dir / "nodes_work" / chunk_dir.name
# #                 dst.mkdir(parents=True, exist_ok=True)
# #                 shutil.copy2(ocr8, dst / "ocr_step8.jsonl")

# #     print("   ✅ intermediates done")

# #     # --- clear others ---
# #     for item in out_dir.iterdir():
# #         if item.name in final_files:
# #             continue
# #         if item.is_dir():
# #             shutil.rmtree(item, ignore_errors=True)
# #         else:
# #             item.unlink(missing_ok=True)

# #     print("✨ 清理完成！")


# # ================================================================
# # 主流程
# # ================================================================
# def main(image_path: str, out_dir: str):
#     image_path = Path(image_path).resolve()
#     out_dir = Path(out_dir).resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\n🚀 AutoBench IMAGE MODE")
#     print(f"Input Image: {image_path}")
#     print(f"Output Dir: {out_dir}")

#     # ============================================================
#     # STEP 1.5 — Layout Detection
#     # ============================================================
#     layout_path = out_dir / "layout.jsonl"
#     if not jsonl_not_empty(layout_path):
#         print("\n=== 🚀 STEP1.5: Layout Detection ===")
#         run([
#             "python",
#             str(BASE / "step1.5_detect_layout.py"),
#             "--img", str(image_path),
#             "--out", str(layout_path)
#         ])
#     else:
#         print("【跳过】STEP1.5 已完成")

#     # ============================================================
#     # STEP 2 — SAM Segment
#     # ============================================================
#     sam_root = out_dir / "sam_results"
#     sam_root.mkdir(exist_ok=True)

#     chunks_bbox = sam_root / "chunks.jsonl"
#     chunks_dir = sam_root / "chunks"

#     if not (jsonl_not_empty(chunks_bbox) and chunks_dir.exists()):
#         print("\n=== 🚀 STEP2: SAM Segment ===")
#         run([
#             "python",
#             str(BASE / "step2_sam_segment.py"),
#             "--img", str(image_path),
#             "--out", str(sam_root),
#             "--ckpt", str(BASE / "sam_vit_h_4b8939.pth")
#         ])
#     else:
#         print("【跳过】STEP2 已完成")

#     # ============================================================
#     # STEP 2.1 — Canvas Size
#     # ============================================================
#     canvas_file = out_dir / "canvas.jsonl"

#     if not file_not_empty(canvas_file):
#         print("\n=== 🚀 STEP2.1: Canvas Size ===")
#         run([
#             "python",
#             str(BASE / "step2.1_compute_canvas_size.py"),
#             "--input", str(chunks_bbox),
#             "--output", str(canvas_file)
#         ])
#     else:
#         print("【跳过】STEP2.1 已完成")

#     # ============================================================
#     # STEP 2.5+2.6 — VLM Title + Summary
#     # ============================================================
#     chunks_summary = sam_root / "chunks_with_titles_and_summary.jsonl"

#     if not jsonl_not_empty(chunks_summary):
#         print("\n=== 🚀 STEP2.5 + 2.6: Title + Summary ===")
#         run([
#             "python",
#             str(BASE / "step2.5_step2.6_chunk_title_and_summary.py"),
#             "--chunks", str(chunks_bbox),
#             "--imgs", str(chunks_dir),
#             "--out", str(chunks_summary)
#         ])
#     else:
#         print("【跳过】STEP2.5+2.6 已完成")

#     # ============================================================
#     # [关键修复] Prepare Chunks Dir for Step 13
#     # 确保 step13 能在 out_dir/chunks 找到文件
#     # ============================================================
#     final_chunks_dir = out_dir / "chunks"
#     final_chunks_dir.mkdir(exist_ok=True)
    
#     # 1. 复制 JSONL
#     if chunks_summary.exists():
#         dst_jsonl = final_chunks_dir / "chunks_with_titles_and_summary.jsonl"
#         # 如果目标不存在或大小不同，则复制
#         if not dst_jsonl.exists() or dst_jsonl.stat().st_size != chunks_summary.stat().st_size:
#             shutil.copy2(chunks_summary, dst_jsonl)
#             print(f"📦 Copied chunks jsonl to {final_chunks_dir}")
    
#     # 2. 复制图片 (chunk0.png ...)
#     if chunks_dir.exists():
#         for png in chunks_dir.glob("chunk*.png"):
#             dst_png = final_chunks_dir / png.name
#             if not dst_png.exists():
#                 shutil.copy2(png, dst_png)

#     # ============================================================
#     # STEP 3 — Best SAM Nodes
#     # ============================================================
#     nodes_sam_root = out_dir / "nodes_sam"
#     nodes_sam_root.mkdir(exist_ok=True)

#     if not any(nodes_sam_root.glob("*_result")):
#         print("\n=== 🚀 STEP3: Best SAM ===")
#         run([
#             "python",
#             str(BASE / "step3_choose_best_sam.py"),
#             "--input", str(chunks_dir),
#             "--output", str(nodes_sam_root)
#         ])
#     else:
#         print("【跳过】STEP3 已完成")

#     # ============================================================
#     # STEP 4 ~ 10 per chunk
#     # ============================================================
#     nodes_work_root = out_dir / "nodes_work"
#     nodes_work_root.mkdir(exist_ok=True)

#     chunk_pngs = sorted(chunks_dir.glob("chunk*.png"))
#     print(f"\n📌 Found {len(chunk_pngs)} chunks")

#     for chunk_img in chunk_pngs:
#         cid = chunk_img.stem
#         work_dir = nodes_work_root / cid
#         work_dir.mkdir(exist_ok=True)

#         nodes_jsonl = nodes_sam_root / (chunk_img.name + "_result") / "nodes.jsonl"

#         print(f"\n=== 🔍 PROCESS CHUNK {cid} ===")

#         # STEP4
#         step4 = work_dir / "nodes_step4.jsonl"
#         if not jsonl_not_empty(step4):
#             run([
#                 "python",
#                 str(BASE / "step4_describe_node_with_type_delete_meaningless.py"),
#                 "--nodes", str(nodes_jsonl),
#                 "--imgs", str(nodes_jsonl.parent / "nodes"),
#                 "--out", str(step4)
#             ])
#         else:
#             print("  【跳过】STEP4")

#         # STEP5
#         step5 = work_dir / "nodes_step5.jsonl"
#         if not jsonl_not_empty(step5):
#             run([
#                 "python",
#                 str(BASE / "step5_delete_sam.py"),
#                 "--nodes", str(step4),
#                 "--out", str(step5)
#             ])
#         else:
#             print("  【跳过】STEP5")

#         # STEP6
#         step6 = work_dir / "nodes_step6.jsonl"
#         if not jsonl_not_empty(step6):
#             run([
#                 "python",
#                 str(BASE / "step6_title_vlm_delete.py"),
#                 "--chunks", str(chunks_summary),
#                 "--nodes", str(step5),
#                 "--out", str(step6)
#             ])
#         else:
#             print("  【跳过】STEP6")

#         # STEP7 (OCR raw)
#         step7 = work_dir / "ocr_raw.jsonl"
#         if not jsonl_not_empty(step7):
#             run([
#                 "python",
#                 str(BASE / "step7_parse_terminal_output.py"),
#                 "--input", str(BASE / "run_dpsk_ocr.py"),
#                 "--img", str(chunk_img),
#                 "--output", str(step7)
#             ])
#         else:
#             print("  【跳过】STEP7")

#         # STEP8
#         step8 = work_dir / "ocr_step8.jsonl"
#         if not jsonl_not_empty(step8):
#             run([
#                 "python",
#                 str(BASE / "step8_filter_title_nodes.py"),
#                 "--nodes", str(step7),
#                 "--titles", str(chunks_summary),
#                 "--out", str(step8)
#             ])
#         else:
#             print("  【跳过】STEP8")

#         # STEP10
#         step10 = work_dir / "nodes_final_local.jsonl"
#         if not jsonl_not_empty(step10):
#             merge2 = step8 if jsonl_not_empty(step8) else step7
#             run([
#                 "python",
#                 str(BASE / "step10_merge.py"),
#                 "--input1", str(step6),
#                 "--input2", str(merge2),
#                 "--output", str(step10)
#             ])
#         else:
#             print("  【跳过】STEP10")

#     # ============================================================
#     # STEP 11 — Mermaid
#     # ============================================================
#     mermaid_root = out_dir / "mermaid"
#     mermaid_root.mkdir(exist_ok=True)

#     if not any(mermaid_root.glob("*.mmd")):
#         print("\n=== 🚀 STEP11: Mermaid ===")
#         env = os.environ.copy()
#         env["DF_MODEL"] = "gemini-2.5-flash-image-preview"
#         run([
#             "python",
#             str(BASE / "step11_chunk_mermaid.py"),
#             "--imgs", str(chunks_dir),
#             "--out", str(mermaid_root)
#         ], env=env)
#     else:
#         print("【跳过】STEP11 已完成")

#     # ============================================================
#     # STEP 12 — Extract Edges
#     # ============================================================
#     for chunk_img in chunk_pngs:
#         cid = chunk_img.stem
#         mmd_file = mermaid_root / f"{cid}.mmd"
#         edge_tmp = nodes_work_root / cid / "nodes_final_local_edges.jsonl"

#         if not mmd_file.exists():
#             print(f"⚠ {cid} 无 mermaid，跳过 Step12")
#             continue

#         if not jsonl_not_empty(edge_tmp):
#             print(f"\n=== 🚀 STEP12: Extract edges for {cid} ===")
#             env = os.environ.copy()
#             env["DF_MODEL"] = "gemini-2.5-flash-image-preview"
#             run([
#                 "python",
#                 str(BASE / "step12_extract_edges.py"),
#                 "--nodes", str(nodes_work_root / cid / "nodes_final_local.jsonl"),
#                 "--mmd", str(mmd_file)
#             ], env=env)
#         else:
#             print(f"  【跳过】STEP12 {cid}")

#     # ============================================================
#     # STEP 12.5 — Global Nodes
#     # ============================================================
#     nodes_dir = out_dir / "nodes"
#     nodes_dir.mkdir(exist_ok=True)

#     global_nodes_file = nodes_dir / "nodes_final.jsonl"

#     if not jsonl_not_empty(global_nodes_file):
#         print("\n=== 🚀 STEP12.5: Build Global Nodes ===")
#         all_nodes = []
#         for chunk_img in chunk_pngs:
#             local = nodes_work_root / chunk_img.stem / "nodes_final_local.jsonl"
#             if jsonl_not_empty(local):
#                 with open(local, "r") as f:
#                     for line in f:
#                         all_nodes.append(json.loads(line))

#         id_map = {}
#         for i, n in enumerate(all_nodes):
#             old = n["node_id"]
#             new = f"node{i}"
#             id_map[old] = new
#             n["node_id"] = new

#         with open(global_nodes_file, "w") as f:
#             for n in all_nodes:
#                 f.write(json.dumps(n, ensure_ascii=False) + "\n")

#         with open(out_dir / "node_id_map.json", "w") as f:
#             json.dump(id_map, f, ensure_ascii=False, indent=2)

#     else:
#         print("【跳过】STEP12.5 已完成")

#     # ============================================================
#     # STEP 13 — Assemble Final JSON
#     # ============================================================
#     canvas_full = out_dir / "canvas_full.json"

#     if not file_not_empty(canvas_full):
#         print("\n=== 🚀 STEP13: Assemble Final JSON ===")
#         run([
#             "python",
#             str(BASE / "step13_assemble_final_json.py"),
#             "--out_dir", str(out_dir),
#             "--output", str(canvas_full)
#         ])
#     else:
#         print("【跳过】STEP13 已完成")

#     # ============================================================
#     # CLEANUP
#     # ============================================================
#     #cleanup_workspace(out_dir)
#     print("\n🎉 ALL DONE.")


# # ============================================================
# # CLI
# # ============================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--img", required=True, help="输入图片路径")
#     parser.add_argument("--out", required=True, help="输出目录")
#     args = parser.parse_args()

#     main(args.img, args.out)