# import os
# import json
# import subprocess
# import argparse
# import shutil
# from pathlib import Path
# from dotenv import load_dotenv
# load_dotenv() 
# BASE = Path(__file__).parent.resolve()

# from pathlib import Path
# from typing import Optional

# def edges_not_empty(path: Path) -> bool:
#     """
#     判断 edges jsonl 是否真的包含边
#     """
#     if not path.exists():
#         return False

#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 obj = json.loads(line)
#                 # 兼容两种格式：{edges:[...]} 或 单条 edge
#                 if isinstance(obj, dict):
#                     if "edges" in obj and len(obj["edges"]) > 0:
#                         return True
#                     if "from" in obj and "to" in obj:
#                         return True
#         return False
#     except Exception:
#         return False


# def resolve_framework_image(pdf_ocr_dir: Path, figure_field: Optional[str]) -> Optional[Path]:
#     """
#     在 run_pipeline 层统一修正 framework_figure 路径问题。
#     支持：
#       - "images/page4_0.jpg"
#       - "page4_0.jpg"
#       - "/abs/path/page4_0.jpg"（兜底）
#     返回：真实存在的 Path，或 None
#     """
#     if not figure_field:
#         return None

#     s = str(figure_field).strip().strip('"').strip("'")

#     # 如果已经是绝对路径
#     p = Path(s)
#     if p.is_absolute() and p.exists():
#         return p

#     # 取 basename（关键）
#     name = p.name

#     # Case 1: pdf_ocr/page4_0.jpg  ← 你现在的真实情况
#     cand1 = pdf_ocr_dir / name
#     if cand1.exists():
#         return cand1

#     # Case 2: pdf_ocr/images/page4_0.jpg
#     cand2 = pdf_ocr_dir / "images" / name
#     if cand2.exists():
#         return cand2

#     # Case 3: 原始相对路径直接拼
#     cand3 = pdf_ocr_dir / s
#     if cand3.exists():
#         return cand3

#     return None

# def run(cmd, env=None):
#     import os
#     print("▶️", " ".join(map(str, cmd)))

#     # inherit caller env first
#     if env is None:
#         env = os.environ.copy()

#     # always inject DF config
#     env["DF_API_KEY"] = os.getenv("DF_API_KEY", "")
#     env["DF_API_URL"] = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
#     if "DF_MODEL" not in env:
#         env["DF_MODEL"] = os.getenv("DF_MODEL", "gpt-4o")

#     subprocess.run(cmd, check=True, env=env)

# def file_not_empty(path: Path):
#     return path.exists() and path.stat().st_size > 10


# def jsonl_not_empty(path: Path):
#     if not path.exists():
#         return False
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 return True
#     return False

# def cleanup_workspace(out_dir: Path):
#     """
#     清理工作区：保留三种最终 canvas 结果以及核心中间件。
#     """
#     print(f"\n🧹 开始整理最终结果到: {out_dir}")
    
#     # === 1. 确定要保留的根目录文件 (新增了三种 canvas) ===
#     final_files = {
#         "method_and_framework.jsonl", 
#         "canvas_full_step6.json",   # 新增
#         "canvas_full_step8.json",   # 新增
#         "canvas_full_merge.json",   # 新增
#         "segmentation_result.png"
#     }

#     # === 2. 移动 sam_results 主分割图到根目录 ===
#     src_seg = out_dir / "sam_results" / "segmentation_result.png"
#     dst_seg = out_dir / "segmentation_result.png"
#     if src_seg.exists():
#         if dst_seg.exists():
#             dst_seg.unlink()
#         shutil.move(str(src_seg), str(dst_seg))
#         print("   ✅ 移动: segmentation_result.png -> /")

#     # === 3. 移动 framework 原始图到根目录 ===
#     ff_json = out_dir / "pdf_ocr" / "merged_result_framework_figure.json"
#     if ff_json.exists():
#         try:
#             with open(ff_json, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 img_name = data.get("framework_figure")
                
#                 if img_name:
#                     src_img = out_dir / "pdf_ocr" / img_name
#                     dst_img = out_dir / img_name
                    
#                     if src_img.exists():
#                         if dst_img.exists():
#                             dst_img.unlink()
#                         shutil.move(str(src_img), str(dst_img))
#                         final_files.add(img_name)
#                         print(f"   ✅ 移动: {img_name} -> /")
#         except Exception as e:
#             print(f"   ⚠️ 读取框架图信息失败: {e}")

#     # === [关键备份] ===
#     intermediates_dir = out_dir / "intermediates"
#     intermediates_dir.mkdir(exist_ok=True)
#     final_files.add("intermediates") 

#     # 备份 1: SAM 子图
#     nodes_sam_src = out_dir / "nodes_sam"
#     if nodes_sam_src.exists():
#         for chunk_dir in nodes_sam_src.glob("*_result"):
#             if chunk_dir.is_dir():
#                 src_file = chunk_dir / "segmentation_result.png"
#                 if src_file.exists():
#                     dst_folder = intermediates_dir / "nodes_sam" / chunk_dir.name
#                     dst_folder.mkdir(parents=True, exist_ok=True)
#                     shutil.copy2(src_file, dst_folder / "segmentation_result.png")

#     # 备份 2: 关键的三个中间 jsonl (step6, step8, step10)
#     # 我们需要在 nodes_work 下面去找
#     nodes_work_src = out_dir / "nodes_work"
#     if nodes_work_src.exists():
#         for chunk_dir in nodes_work_src.iterdir():
#             if chunk_dir.is_dir():
#                 dst_folder = intermediates_dir / "nodes_work" / chunk_dir.name
#                 dst_folder.mkdir(parents=True, exist_ok=True)
                
#                 # 备份我们要对比的三种文件
#                 for fname in ["nodes_step6.jsonl", "ocr_step8.jsonl", "nodes_final_local.jsonl"]:
#                     src_f = chunk_dir / fname
#                     if src_f.exists():
#                         shutil.copy2(src_f, dst_folder / fname)

#     print(f"   ✅ 已备份关键中间文件到: intermediates/")

#     # === 4. 暴力清理 ===
#     for item in out_dir.iterdir():
#         if item.name in final_files:
#             continue
#         if item.is_dir():
#             try:
#                 shutil.rmtree(item)
#             except Exception as e:
#                 print(f"   ❌ 删除文件夹失败 {item.name}: {e}")
#         else:
#             try:
#                 item.unlink()
#             except Exception as e:
#                 print(f"   ❌ 删除文件失败 {item.name}: {e}")

#     print("✨ 清理完成！目录已整理。")

# def main(pdf_path: str, out_dir: str):
#     pdf_path = Path(pdf_path).resolve()
#     out_dir = Path(out_dir).resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # ... (Step 0 到 Step 5 保持不变) ...
#     # ------------------------------
#     # 0. OCR
#     # ------------------------------
#     pdf_ocr_dir = out_dir / "pdf_ocr"
#     pdf_ocr_dir.mkdir(exist_ok=True)
#     md_path = pdf_ocr_dir / "merged_result.md"

#     if not file_not_empty(md_path):
#         print("\n=== 🚀 STEP0: OCR Running ===")
#         env = os.environ.copy()
#         env["PDF_FILE"] = str(pdf_path)
#         env["OUTPUT_DIR"] = str(pdf_ocr_dir)
#         run(["python", str(BASE / "step0_run_pdf_infer_ocr.py")], env=env)
#         if not file_not_empty(md_path):
#             raise RuntimeError("OCR failed: merged_result.md missing.")
#     else:
#         print("【跳过】STEP0 OCR 已完成")

#     # ------------------------------
#     # 1. Select Framework Figure
#     # ------------------------------
#     ff_json = pdf_ocr_dir / "merged_result_framework_figure.json"
#     if not file_not_empty(ff_json):
#         print("\n=== 🚀 STEP1: Framework Selector ===")
#         run(["python",
#              str(BASE / "step1_framework_figure_selector.py"),
#              "--md", str(md_path)
#         ])
#     else:
#         print("【跳过】STEP1 框架图选择 已完成")

#     with open(ff_json, "r", encoding="utf-8") as f:
#         ff_info = json.load(f)
#     # framework_name = ff_info.get("framework_figure")
#     # framework_img = pdf_ocr_dir / framework_name

#     # if not framework_img.exists():
#     #     print(f"❌ Framework image not found: {framework_img}. Stopping the pipeline.")
#     #     return
#     framework_name = ff_info.get("framework_figure")

#     framework_img = resolve_framework_image(pdf_ocr_dir, framework_name)

#     if framework_img is None:
#         print(f"❌ Framework image not found (raw={framework_name}), stopping.")
#         return

#     print(f"✅ Resolved framework image: {framework_img}")

#     # ------------------------------
#     # 1.2 Extract Method and Figure Titles
#     # ------------------------------
#     # print("\n=== 🚀 STEP1.2: Extract Method and Figure Titles ===")
#     # method_and_figure_jsonl = out_dir / "method_and_framework.jsonl"
#     # if not jsonl_not_empty(method_and_figure_jsonl):
#     #     run(["python", 
#     #          str(BASE / "step1.2_extract_method_and_figure_titles.py"),
#     #          "--md", str(md_path),
#     #          "--out", str(method_and_figure_jsonl)
#     #     ])
#     # else:
#     #     print("【跳过】STEP1.2 Method and Figure Titles Extraction 已完成")

#     # ------------------------------
#     # 1.5 Layout Detection
#     # ------------------------------
#     layout_path = out_dir / "layout.jsonl"
#     if not jsonl_not_empty(layout_path):
#         print("\n=== 🚀 STEP1.5: Layout Detection ===")
#         run(["python",
#              str(BASE / "step1.5_detect_layout.py"),
#              "--img", str(framework_img),
#              "--out", str(layout_path)
#         ])
#     else:
#         print("【跳过】STEP1.5 Layout 已完成")

#     # ------------------------------
#     # 2. SAM Chunk Segmentation
#     # ------------------------------
#     sam_root = out_dir / "sam_results"
#     sam_root.mkdir(exist_ok=True)
#     chunks_bbox_jsonl = sam_root / "chunks.jsonl"
#     chunks_img_dir = sam_root / "chunks"

#     if not (jsonl_not_empty(chunks_bbox_jsonl) and chunks_img_dir.exists()):
#         print("\n=== 🚀 STEP2: SAM Segment ===")
#         run(["python",
#              str(BASE / "step2_sam_segment.py"),
#              "--img", str(framework_img),
#              "--out", str(sam_root),
#              "--ckpt", str(BASE / "sam_vit_h_4b8939.pth")
#         ])
#     else:
#         print("【跳过】STEP2 SAM 分块 已完成")

#     # ------------------------------
#     # 2.1 Canvas Size
#     # ------------------------------
#     canvas_path = out_dir / "canvas.jsonl"
#     if not file_not_empty(canvas_path):
#         print("\n=== 🚀 STEP2.1: Canvas Size ===")
#         run(["python",
#              str(BASE / "step2.1_compute_canvas_size.py"),
#              "--input", str(chunks_bbox_jsonl),
#              "--output", str(canvas_path)
#         ])
#     else:
#         print("【跳过】STEP2.1 Canvas 已完成")

#     # ------------------------------
#     # 2.5 + 2.6 Unified: Chunk Titles + Summary
#     # ------------------------------
#     chunks_with_titles_and_summary = sam_root / "chunks_with_titles_and_summary.jsonl"

#     if not jsonl_not_empty(chunks_with_titles_and_summary):
#         print("\n=== 🚀 STEP2.5 + STEP2.6: Title + Summary (Unified) ===")
#         run([
#             "python",
#             str(BASE / "step2.5_step2.6_chunk_title_and_summary.py"),
#             "--chunks", str(chunks_bbox_jsonl),
#             "--imgs", str(chunks_img_dir),
#             "--out", str(chunks_with_titles_and_summary)
#         ])
#     else:
#         print("【跳过】STEP2.5+2.6 Title+Summary 已完成")

#     # 拷贝最终 chunks
#     final_chunks_dir = out_dir / "chunks"
#     final_chunks_dir.mkdir(exist_ok=True)
#     for png in chunks_img_dir.glob("chunk*.png"):
#         dst = final_chunks_dir / png.name
#         if not dst.exists():
#             shutil.copy2(png, dst)
#     shutil.copy2(chunks_with_titles_and_summary,
#                  final_chunks_dir / "chunks_with_titles_and_summary.jsonl")

#     # ------------------------------
#     # 3. Choose Best SAM nodes
#     # ------------------------------
#     nodes_sam_root = out_dir / "nodes_sam"
#     nodes_sam_root.mkdir(exist_ok=True)

#     if not (nodes_sam_root.exists() and any(nodes_sam_root.glob("*_result"))):
#         print("\n=== 🚀 STEP3: Choose Best SAM ===")
#         run(["python",
#              str(BASE / "step3_choose_best_sam.py"),
#              "--input", str(chunks_img_dir),
#              "--output", str(nodes_sam_root)
#         ])
#     else:
#         print("【跳过】STEP3 Best SAM 已完成")

#     # ------------------------------
#     # 4. Node processing (loop per chunk)
#     # ------------------------------
#     nodes_work_root = out_dir / "nodes_work"
#     nodes_work_root.mkdir(exist_ok=True)

#     chunk_pngs = sorted(chunks_img_dir.glob("chunk*.png"))
#     print(f"📌 Found {len(chunk_pngs)} chunks")

#     for chunk_img in chunk_pngs:
#         chunk_id = chunk_img.stem
#         work_dir = nodes_work_root / chunk_id
#         work_dir.mkdir(exist_ok=True)

#         print(f"\n=== 🔍 PROCESS CHUNK {chunk_id} ===")

#         nodes_jsonl = nodes_sam_root / f"{chunk_img.name}_result" / "nodes.jsonl"

#         # Step4
#         step4_out = work_dir / "nodes_step4.jsonl"
#         if not jsonl_not_empty(step4_out):
#             run(["python",
#                  str(BASE / "step4_describe_node_with_type_delete_meaningless.py"),
#                  "--nodes", str(nodes_jsonl),
#                  "--imgs", str(nodes_jsonl.parent / "nodes"),
#                  "--out", str(step4_out)
#             ])
#         else:
#             print(f"【跳过】{chunk_id} STEP4 已完成")

#         # Step5
#         step5_out = work_dir / "nodes_step5.jsonl"
#         if not jsonl_not_empty(step5_out):
#             run(["python",
#                  str(BASE / "step5_delete_sam.py"),
#                  "--nodes", str(step4_out),
#                  "--out", str(step5_out)
#             ])
#         else:
#             print(f"【跳过】{chunk_id} STEP5 已完成")

#         # Step6 (Output: nodes_step6.jsonl) -> 路径A源头
#         step6_out = work_dir / "nodes_step6.jsonl"
#         if not jsonl_not_empty(step6_out):
#             run(["python",
#                  str(BASE / "step6_title_vlm_delete.py"),
#                  "--chunks", str(chunks_with_titles_and_summary),
#                  "--nodes", str(step5_out),
#                  "--out", str(step6_out)
#             ])
#         else:
#             print(f"【跳过】{chunk_id} STEP6 已完成")

#         # Step7 OCR
#         step7_raw = work_dir / "ocr_raw.jsonl"
#         if not jsonl_not_empty(step7_raw):
#             run(["python",
#                  str(BASE / "step7_parse_terminal_output.py"),
#                  "--input", str(BASE / "run_dpsk_ocr.py"),
#                  "--img", str(chunk_img),
#                  "--output", str(step7_raw)
#             ])
#         else:
#             print(f"【跳过】{chunk_id} STEP7 已完成")

#         # Step8 (Output: ocr_step8.jsonl) -> 路径B源头
#         step8_out = work_dir / "ocr_step8.jsonl"
#         if not jsonl_not_empty(step8_out):
#             run(["python",
#                  str(BASE / "step8_filter_title_nodes.py"),
#                  "--nodes", str(step7_raw),
#                  "--titles", str(chunks_with_titles_and_summary),
#                  "--out", str(step8_out)
#             ])
#         else:
#             print(f"【跳过】{chunk_id} STEP8 已完成")

#         # Step10 Merge (Output: nodes_final_local.jsonl) -> 路径C源头
#         step10_out = work_dir / "nodes_final_local.jsonl"
#         if not jsonl_not_empty(step10_out):
#             if step8_out.exists() and jsonl_not_empty(step8_out):
#                 merge_input2 = step8_out
#             else:
#                 print(f"【注意】{chunk_id} Step8 结果缺失或为空，Step10 将使用 Step7 原始数据")
#                 merge_input2 = step7_raw
#             run(["python",
#                  str(BASE / "step10_merge.py"),
#                  "--input1", str(step6_out),
#                  "--input2", str(merge_input2),
#                  "--output", str(step10_out)
#             ])
#         else:
#             print(f"【跳过】{chunk_id} STEP10 已完成")

#     # ------------------------------
#     # 5. Local mermaid (共用)
#     # ------------------------------
#     mermaid_root = out_dir / "mermaid"
#     mermaid_root.mkdir(exist_ok=True)

#     if not any(mermaid_root.glob("*.mmd")):
#         print("\n=== 🚀 STEP11: Mermaid (Shared) ===")
#         env = os.environ.copy()
#         env["DF_MODEL"] = "gemini-2.5-flash-image-preview"
#         run(["python",
#             str(BASE / "step11_chunk_mermaid.py"),
#             "--imgs", str(chunks_img_dir),
#             "--out", str(mermaid_root)
#         ], env=env)
#     else:
#         print("【跳过】STEP11 Mermaid 已完成")

#     # =========================================================================
#     # 🚀 MULTI-TRACK PIPELINE: Step 12 -> 13
#     # 针对三种源文件分别执行 Edge Extraction 和 Assembly
#     # =========================================================================

#     # 定义三条处理轨道
#     # name: 最终输出的后缀 (canvas_full_{name}.json)
#     # file_name: 对应的源节点文件名
#     tracks = [
#         {"name": "step6",  "file_name": "nodes_step6.jsonl"},
#         {"name": "step8",  "file_name": "ocr_step8.jsonl"},
#         {"name": "merge",  "file_name": "nodes_final_local_backup.jsonl"} 
#     ]

#     print("\n⚡️ 启动多轨道处理: [step6, step8, merge]")
    
#     # 1. 先备份 Step 10 生成的原始 nodes_final_local.jsonl
#     for chunk_img in chunk_pngs:
#         chunk_id = chunk_img.stem
#         work_dir = nodes_work_root / chunk_id
#         original_merge = work_dir / "nodes_final_local.jsonl"
#         backup_merge = work_dir / "nodes_final_local_backup.jsonl"
        
#         # 强制更新备份（如果文件存在）
#         if original_merge.exists():
#             shutil.copy2(original_merge, backup_merge)

#     nodes_dir = out_dir / "nodes"
#     nodes_dir.mkdir(exist_ok=True)
#     images_dir = nodes_dir / "images"
#     images_dir.mkdir(exist_ok=True)

#     # ------------------------------
#     # 循环执行三个轨道
#     # ------------------------------
#     for track in tracks:
#         track_name = track["name"]
#         source_file_name = track["file_name"]
        
#         print(f"\n>>> 🛤️  Running Track: {track_name} (Source: {source_file_name})")

#         # === A. 伪装文件 & 运行 Step 12 (Edges) ===
#         for chunk_img in chunk_pngs:
#             chunk_id = chunk_img.stem
#             work_dir = nodes_work_root / chunk_id
            
#             mmd_file = mermaid_root / f"{chunk_id}.mmd"
#             # 如果没有 mermaid 文件，说明无法提取边，但我们仍需生成点
#             has_mmd = (mmd_file.exists() and file_not_empty(mmd_file))
#             if not has_mmd:
#                  print(f"   ⚠ {chunk_id}: 无 mermaid，后续将生成无边图")

#             target_file = work_dir / "nodes_final_local.jsonl"     # 目标：被伪装的文件
#             source_file = work_dir / source_file_name              # 源：当前轨道的数据
            
#             # 1. 覆盖伪装
#             if source_file.exists():
#                 # [安全检查] 只有当源和目标不是同一个文件时才复制
#                 if source_file.resolve() != target_file.resolve():
#                     shutil.copy2(source_file, target_file)
                
#                 # =========================================================
#                 # 🔧 [HOT FIX] 补丁：确保 node_id 存在
#                 # =========================================================
#                 patched_nodes = []
#                 needs_rewrite = False
                
#                 try:
#                     with open(target_file, "r", encoding="utf-8") as f:
#                         for idx, line in enumerate(f):
#                             line = line.strip()
#                             if not line: continue
#                             try:
#                                 node = json.loads(line)
#                                 # 检查并修复 node_id
#                                 if "node_id" not in node:
#                                     node["node_id"] = f"track_{track_name}_{idx}"
#                                     needs_rewrite = True
                                
#                                 # 顺便检查 bbox
#                                 if "bbox" not in node:
#                                     node["bbox"] = [0, 0, 0, 0]
#                                     needs_rewrite = True
                                    
#                                 patched_nodes.append(node)
#                             except json.JSONDecodeError:
#                                 continue
                    
#                     if needs_rewrite:
#                         print(f"   🔧 Patching missing 'node_id' for {track_name} in {chunk_id}...")
#                         with open(target_file, "w", encoding="utf-8") as f:
#                             for node in patched_nodes:
#                                 f.write(json.dumps(node, ensure_ascii=False) + "\n")
#                 except Exception as e:
#                     print(f"   ⚠️ Patching failed: {e}")
#                 # =========================================================

#             else:
#                 print(f"   ⚠️ Warning: Source {source_file_name} missing for {chunk_id}, using empty.")
#                 with open(target_file, 'w') as f: pass

#             # 2. 运行 Step 12 (生成 edges)
#             edges_file = work_dir / "nodes_final_local_edges.jsonl"
#             MAX_RETRY = 5

#             if has_mmd:
#                 env = os.environ.copy()
#                 env["DF_MODEL"] = "gemini-2.5-flash-image-preview"

#                 success = False
#                 for attempt in range(1, MAX_RETRY + 1):
#                     print(f"   🔁 Step12 attempt {attempt}/{MAX_RETRY} for {chunk_id} ({track_name})")

#                     # 每次重试前清空旧 edges
#                     if edges_file.exists():
#                         edges_file.unlink()

#                     try:
#                         run([
#                             "python",
#                             str(BASE / "step12_extract_edges.py"),
#                             "--nodes", str(target_file),
#                             "--mmd", str(mmd_file)
#                         ], env=env)
#                     except subprocess.CalledProcessError:
#                         pass

#                     # ✅ 核心判断：是不是真的有边
#                     if edges_not_empty(edges_file):
#                         print(f"   ✅ Edges generated successfully for {chunk_id}")
#                         success = True
#                         break
#                     else:
#                         print(f"   ⚠️ No valid edges, retrying...")

#                 if not success:
#                     print(f"   ❌ Step12 failed after {MAX_RETRY} retries, fallback to empty edges")
#                     with open(edges_file, "w") as f:
#                         pass
#             else:
#                 # 没有 mermaid，也要保证 edges 文件存在
#                 with open(work_dir / "nodes_final_local_edges.jsonl", "w") as f:
#                     pass

        
        
#         # === B. 运行 Step 12.5 (Global Nodes) ===
#         # 这会读取所有 chunk 的 nodes_final_local.jsonl (当前是伪装版)
#         # 并生成 out_dir/nodes/nodes_final.jsonl
#         print(f"\n   Running Step12.5 (Global Build) for {track_name}...")
        
#         # 这里的逻辑直接内嵌，不再依赖外部脚本，确保完全控制
#         all_nodes = []
#         for chunk_img in chunk_pngs:
#             chunk_id = chunk_img.stem
#             local = nodes_work_root / chunk_id / "nodes_final_local.jsonl"
#             if jsonl_not_empty(local):
#                 with open(local, "r", encoding="utf-8") as f:
#                     for line in f:
#                         node = json.loads(line.strip())
#                         all_nodes.append(node)
        
#         # 重新编号
#         id_map = {}
#         for i, n in enumerate(all_nodes):
#             old = n["node_id"]
#             new = f"node{i}"
#             id_map[old] = new
#             n["node_id"] = new
        
#         # 写入全局 nodes_final.jsonl (会被 Step 13 读取)
#         global_nodes_file = nodes_dir / "nodes_final.jsonl"
#         with open(global_nodes_file, "w", encoding="utf-8") as f:
#             for n in all_nodes:
#                 f.write(json.dumps(n, ensure_ascii=False) + "\n")
        
#         # 写入 ID map (会被 Step 13 读取)
#         with open(out_dir / "node_id_map.json", "w", encoding="utf-8") as f:
#             json.dump(id_map, f, ensure_ascii=False, indent=2)

#         # === C. 运行 Step 13 (Assemble) ===
#         print(f"\n   Running Step13 (Assemble) for {track_name}...")
#         # Step 13 读取 out_dir 下的 nodes/nodes_final.jsonl 和 node_id_map.json
#         # 以及各个 chunk 下的 nodes_final_local_edges.jsonl (Step 12 刚生成的)
#         temp_canvas = out_dir / "canvas_full.json"
        
#         run(["python",
#              str(BASE / "step13_assemble_final_json.py"),
#              "--out_dir", str(out_dir),
#              "--output", str(temp_canvas)
#         ])

#         # === D. 保存结果 ===
#         final_track_output = out_dir / f"canvas_full_{track_name}.json"
#         if temp_canvas.exists():
#             shutil.move(str(temp_canvas), str(final_track_output))
#             print(f"✅ Generated: {final_track_output.name}")
#         else:
#             print(f"❌ Failed to generate canvas for {track_name}")

#     # ------------------------------
#     # 恢复现场
#     # ------------------------------
#     print("\n🔄 Restoring original files...")
#     for chunk_img in chunk_pngs:
#         chunk_id = chunk_img.stem
#         work_dir = nodes_work_root / chunk_id
#         original_merge = work_dir / "nodes_final_local.jsonl"
#         backup_merge = work_dir / "nodes_final_local_backup.jsonl"
        
#         if backup_merge.exists():
#             if original_merge.exists():
#                 original_merge.unlink()
#             shutil.move(str(backup_merge), str(original_merge))

#     # 为了保持原本的文件结构一致性，我们可以把 canvas_full_merge.json 复制一份为 canvas_full.json
#     # 这样用户如果只看默认结果也能找到
#     if (out_dir / "canvas_full_merge.json").exists():
#         shutil.copy2(out_dir / "canvas_full_merge.json", out_dir / "canvas_full.json")

#     # ------------------------------
#     # Cleanup
#     # ------------------------------
#     #cleanup_workspace(out_dir)

#     print("\n🎉 ALL DONE.")
#     print(f"Output root: {out_dir}")
#     print("Files created:")
#     print("  - canvas_full_step6.json (Only VLM)")
#     print("  - canvas_full_step8.json (Only OCR)")
#     print("  - canvas_full_merge.json (Merged)")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pdf", required=True, help="输入 PDF 路径")
#     parser.add_argument("--out", required=True, help="输出根目录，比如 output/")
#     args = parser.parse_args()

#     main(args.pdf, args.out)


import os
import json
import subprocess
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
BASE = Path(__file__).parent.resolve()

# ============================================================
# Utils
# ============================================================
def ensure_node_ids(path: Path, cid: str, track: str):
    """
    检查文件中的节点是否有 node_id。
    如果没有，生成稳定的 node_id 并覆盖写入原文件。
    返回：True (表示文件存在且有效), False (文件不存在)
    """
    if not path.exists():
        return False
    
    lines = []
    needs_rewrite = False
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                n = json.loads(line)
                # 检查 node_id
                if "node_id" not in n:
                    n["node_id"] = f"{cid}_{track}_{i}" # 生成稳定ID
                    needs_rewrite = True
                
                # 顺手检查 bbox，防止报错
                if "bbox" not in n:
                    n["bbox"] = [0, 0, 0, 0]
                    needs_rewrite = True
                    
                lines.append(n)
            except:
                continue

    # 只有当确实修改了内容时，才回写文件
    if needs_rewrite:
        print(f"   🔧 Patching missing IDs in {path.name}...")
        with open(path, "w", encoding="utf-8") as f:
            for n in lines:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
                
    return True

def run(cmd, env=None):
    print("▶️", " ".join(map(str, cmd)))
    if env is None:
        env = os.environ.copy()

    env["DF_API_KEY"] = os.getenv("DF_API_KEY")
    env["DF_API_URL"] = os.getenv("DF_API_URL", "http://localhost:3888/v1")
    if "DF_MODEL" not in env:
        env["DF_MODEL"] = "gpt-4o"

    subprocess.run(cmd, check=True, env=env)


def file_not_empty(path: Path):
    return path.exists() and path.stat().st_size > 10


def jsonl_not_empty(path: Path):
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8") as f:
        return any(line.strip() for line in f)


def edges_not_empty(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    if "edges" in obj and obj["edges"]:
                        return True
                    if "from" in obj and "to" in obj:
                        return True
        return False
    except Exception:
        return False


def resolve_framework_image(pdf_ocr_dir: Path, figure_field: Optional[str]) -> Optional[Path]:
    if not figure_field:
        return None
    s = str(figure_field).strip().strip('"').strip("'")
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p
    for cand in [pdf_ocr_dir / p.name, pdf_ocr_dir / "images" / p.name, pdf_ocr_dir / s]:
        if cand.exists():
            return cand
    return None


def patch_nodes_tmp(src: Path, dst: Path, cid: str, track: str):
    """(可选) 辅助函数：如果源文件缺失 node_id/bbox，生成一个临时文件修复它"""
    patched = []
    if not src.exists():
        return False
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            n = json.loads(line)
            # 补丁：有些中间步骤可能没写 node_id
            n.setdefault("node_id", f"{cid}_{track}_{i}")
            n.setdefault("bbox", [0, 0, 0, 0])
            patched.append(n)
    with open(dst, "w", encoding="utf-8") as f:
        for n in patched:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    return True


# ============================================================
# Main
# ============================================================

def main(pdf_path: str, out_dir: str):
    pdf_path = Path(pdf_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- STEP0 OCR ----------------
    pdf_ocr = out_dir / "pdf_ocr"
    pdf_ocr.mkdir(exist_ok=True)
    md = pdf_ocr / "merged_result.md"

    if not file_not_empty(md):
        env = os.environ.copy()
        env["PDF_FILE"] = str(pdf_path)
        env["OUTPUT_DIR"] = str(pdf_ocr)
        run(["python", str(BASE / "step0_run_pdf_infer_ocr.py")], env)

    # ---------------- STEP1 Framework ----------------
    ff_json = pdf_ocr / "merged_result_framework_figure.json"
    if not file_not_empty(ff_json):
        run(["python", str(BASE / "step1_framework_figure_selector.py"), "--md", str(md)])

    ff = json.loads(ff_json.read_text())
    framework_img = resolve_framework_image(pdf_ocr, ff.get("framework_figure"))
    if framework_img is None:
        raise RuntimeError("Framework image not found")

    # ---------------- STEP2 SAM ----------------
    sam_root = out_dir / "sam_results"
    sam_root.mkdir(exist_ok=True)
    chunks_img_dir = sam_root / "chunks"
    chunks_jsonl = sam_root / "chunks.jsonl"

    if not (jsonl_not_empty(chunks_jsonl) and chunks_img_dir.exists()):
        run([
            "python", str(BASE / "step2_sam_segment.py"),
            "--img", str(framework_img),
            "--out", str(sam_root),
            "--ckpt", str(BASE / "sam_vit_h_4b8939.pth")
        ])

    # ---------------- STEP2.1 Canvas ----------------
    canvas_jsonl = out_dir / "canvas.jsonl"
    if not file_not_empty(canvas_jsonl):
        run([
            "python", str(BASE / "step2.1_compute_canvas_size.py"),
            "--input", str(chunks_jsonl),
            "--output", str(canvas_jsonl)
        ])

    # ---------------- STEP3 Best SAM ----------------
    nodes_sam = out_dir / "nodes_sam"
    nodes_sam.mkdir(exist_ok=True)
    if not any(nodes_sam.glob("*_result")):
        run([
            "python", str(BASE / "step3_choose_best_sam.py"),
            "--input", str(chunks_img_dir),
            "--output", str(nodes_sam)
        ])

    # ---------------- STEP4–5–7–10 (Chunk 独立流程) ----------------
    nodes_work = out_dir / "nodes_work"
    nodes_work.mkdir(exist_ok=True)
    chunk_imgs = sorted(chunks_img_dir.glob("chunk*.png"))

    for chunk_img in chunk_imgs:
        cid = chunk_img.stem
        work = nodes_work / cid
        work.mkdir(exist_ok=True)
        if (work / ".chunk_done").exists():
            continue

        nodes_jsonl = nodes_sam / f"{chunk_img.name}_result" / "nodes.jsonl"

        step4 = work / "nodes_step4.jsonl"
        if not jsonl_not_empty(step4):
            run([
                "python", str(BASE / "step4_describe_node_with_type_delete_meaningless.py"),
                "--nodes", str(nodes_jsonl),
                "--imgs", str(nodes_jsonl.parent / "nodes"),
                "--out", str(step4)
            ])

        step5 = work / "nodes_step5.jsonl"
        if not jsonl_not_empty(step5):
            run(["python", str(BASE / "step5_delete_sam.py"), "--nodes", str(step4), "--out", str(step5)])

        step7 = work / "ocr_raw.jsonl"
        if not jsonl_not_empty(step7):
            run([
                "python", str(BASE / "step7_parse_terminal_output.py"),
                "--input", str(BASE / "run_dpsk_ocr.py"),
                "--img", str(chunk_img),
                "--output", str(step7)
            ])

        step10 = work / "nodes_final_local.jsonl"
        if not jsonl_not_empty(step10):
            run([
                "python", str(BASE / "step10_merge.py"),
                "--input1", str(step5),
                "--input2", str(step7),
                "--output", str(step10)
            ])

        (work / ".chunk_done").write_text("ok")

    # ---------------- STEP11 Mermaid（只跑一次） ----------------
    mermaid_root = out_dir / "mermaid"
    mermaid_root.mkdir(exist_ok=True)
    if not any(mermaid_root.glob("*.mmd")):
        env = os.environ.copy()
        env["DF_MODEL"] = "gemini-2.5-flash-image-preview"
        run([
            "python", str(BASE / "step11_chunk_mermaid.py"),
            "--imgs", str(chunks_img_dir),
            "--out", str(mermaid_root)
        ], env)

    # ========================================================
    # MULTI-TRACK (CLEAN VERSION: NO OVERWRITING)
    # ========================================================
    tracks = [
        ("step5",  "nodes_step5.jsonl"),
        ("step7",  "ocr_raw.jsonl"),
        ("step10", "nodes_final_local.jsonl"), 
    ]

    for track, node_fname in tracks:
        out_canvas = out_dir / f"canvas_full_{track}.json"
        
        if file_not_empty(out_canvas):
            continue

        print(f"\n🚀 Running Track: {track} (Nodes: {node_fname})")

        # ---- Step 12: 生成特定 track 的连线 ----
        for chunk_img in chunk_imgs:
            cid = chunk_img.stem
            work = nodes_work / cid
            
            # 1. 确定源节点文件
            node_src = work / node_fname
            
            # 【核心修复】确保源文件里有 node_id，没有就现场补上并保存
            if not ensure_node_ids(node_src, cid, track):
                continue

            # 2. 确定目标边文件
            edges_dst = work / f"edges_{track}.jsonl"
            
            # 如果边已经有了，跳过
            if edges_not_empty(edges_dst):
                continue

            mmd = mermaid_root / f"{cid}.mmd"
            if not mmd.exists():
                edges_dst.write_text("") 
                continue

            # 3. 运行 Step 12 (现在可以直接用 node_src 了，因为 ID 肯定存在)
            env = os.environ.copy()
            env["DF_MODEL"] = "gemini-3-pro-image-preview"
            
            # 注意：step12 内部通过 --nodes 读取文件，
            # 现在 node_src 里的 ID 和稍后 step13 读取的一模一样
            run([
                "python", str(BASE / "step12_extract_edges.py"),
                "--nodes", str(node_src),
                "--mmd", str(mmd),
                "--output", str(edges_dst)
            ], env)

        # ---- Step 13: 组装 ----
        tmp_final = out_dir / "canvas_full.json"
        
        run([
            "python", str(BASE / "step13_assemble_final_json.py"),
            "--out_dir", str(out_dir),
            "--output", str(tmp_final),
            "--nodes_file", node_fname,        
            "--edges_file", f"edges_{track}.jsonl"
        ])
        
        shutil.move(tmp_final, out_canvas)
        print(f"✅ Generated: canvas_full_{track}.json")
    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pdf, args.out)