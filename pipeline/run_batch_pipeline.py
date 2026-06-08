import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# Run a single PDF with your full pipeline
# ============================================================
def run_single_pdf(pdf_path: Path, out_root: Path):
    """
    调用你主 pipeline.py 对单个 PDF 进行处理。
    采用 subprocess 独立进程运行，互不影响。
    """

    pdf_name = pdf_path.stem
    out_dir = out_root / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 判断是否已完成：判断 canvas_full.json 是否存在且大小正常
    final_json_merge = out_dir / "canvas_full_merge.json"
    final_json_step10 = out_dir / "canvas_full_step10.json"

    # 只要有一个存在且不为空，就视为已完成
    if (final_json_merge.exists() and final_json_merge.stat().st_size > 10) or \
       (final_json_step10.exists() and final_json_step10.stat().st_size > 10):
        print(f"✔ 跳过（已完成）: {pdf_name}")
        return

    print(f"\n🚀 开始处理: {pdf_name}")

    cmd = [
        "python",
        str(BASE_DIR / "run_pipeline.py"),
        "--pdf", str(pdf_path),
        "--out", str(out_dir)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"🎉 完成: {pdf_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {pdf_name} — {e}")


# ============================================================
# Batch run wrapper（可选并行）
# ============================================================
def batch_run(pdf_dir: Path, out_root: Path, num_workers: int = 1):
    """
    批量执行 pipeline.py，对每个 PDF 单独处理。
    支持并行 & 断点续跑。
    """
    pdf_list = sorted([p for p in pdf_dir.glob("*.pdf")])
    print(f"📌 发现 {len(pdf_list)} 个 PDF")

    if num_workers == 1:
        # --- 单进程模式 ---
        for pdf in pdf_list:
            run_single_pdf(pdf, out_root)
    else:
        # --- 多进程并行模式 ---
        print(f"⚙️ 使用 {num_workers} 并行 worker")

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = { ex.submit(run_single_pdf, pdf, out_root): pdf for pdf in pdf_list }

            for fut in as_completed(futures):
                pdf = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"❌ 异常: {pdf.name} — {e}")

    print("\n==========================")
    print("🎉 所有任务完成")
    print("==========================")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run pipeline on many PDFs")
    parser.add_argument("--pdf_dir", required=True, help="包含 PDF 的目录，例如 1k_arxiv/")
    parser.add_argument("--out_root", required=True, help="输出根目录")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行 worker 数（默认单进程）")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    batch_run(pdf_dir, out_root, num_workers=args.workers)

# python AutoBench/dataflow_agent/bench/batch_run_pipeline.py --pdf_dir nlp_2025 --out_root nlp_2025_autobench_missing_canvas