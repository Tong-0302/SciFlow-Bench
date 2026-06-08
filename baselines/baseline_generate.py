import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import glob
import argparse
import asyncio
import subprocess
import shutil
import gc
import torch
import time
from pathlib import Path
from typing import List, Optional, Set

# ===== baseline modules =====
try:
    import Graphviz as GZ
    import sdxl
    import pixart
    import flux
    import flux2
    import qwen
    import seedream
    import gemini_dall_e as GD
except ImportError as e:
    print(f"⚠️ [Import Error] Missing modules: {e}")

# ===== PREPARE modules =====
try:
    from pipeline.prepare_prompts import (
        extract_pdf_text,
        run_paper_idea_extractor,
        generate_fig_desc,
    )
except ImportError as e:
    print(f"⚠️ [Import Error] Missing modules: {e}")
    # 打印详细的错误信息，查看具体是哪个模块缺失
    import sys
    print("Python Path:", sys.path)  # 打印当前的 Python 模块查找路径
    print("Ensure that the 'prepare_prompts.py' file is in the correct location.")


# ============================================================
# Robust Utils (Breakpoint Support)
# ============================================================

def is_file_valid(path: Path, min_size: int = 100) -> bool:
    """
    断点续跑的核心：
    1. 文件存在
    2. 文件大小 > min_size (避免崩溃留下的 0kb 空文件)
    """
    if not path.exists():
        return False
    if path.stat().st_size < min_size:
        print(f"⚠️ [Corrupt Found] Removing incomplete file: {path.name}")
        path.unlink() # 删除坏文件，以便重跑
        return False
    return True

def is_bench_valid(out_dir: Path) -> bool:
    """检查 AutoBench 是否真正完成 (JSON 必须合法)"""
    p = out_dir / "canvas_full.json"
    if not p.exists():
        return False
    
    # 尝试读取，如果 JSON 损坏说明上次挂了，删掉重跑
    try:
        if p.stat().st_size < 10: return False
        with open(p, 'r') as f:
            json.load(f)
        return True
    except:
        print(f"⚠️ [Corrupt Bench] Removing invalid JSON in {out_dir.name}")
        p.unlink()
        return False

# ============================================================
# Basic Utils
# ============================================================
def organize_raw_pdfs(root: Path):
    if not root.exists(): return
    pdf_files = [f for f in root.glob("*.pdf") if f.is_file() and f.name != "paper.pdf"]
    for pdf in pdf_files:
        try:
            target_dir = root / pdf.stem
            target_dir.mkdir(exist_ok=True)
            target_file = target_dir / "paper.pdf"
            if not target_file.exists():
                shutil.move(str(pdf), str(target_file))
        except: pass

def find_paper_dirs(root: Path) -> List[Path]:
    # 只要有 paper.pdf 就算有效目录
    return sorted([Path(p) for p in glob.glob(str(root / "*")) if (Path(p) / "paper.pdf").exists()])

def load_prompt_text(prompt_json: Path) -> str:
    with open(prompt_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prompt"]

def run_subprocess_sync(cmd: List[str]):
    env = os.environ.copy()
    env["DF_API_KEY"] = os.getenv("DF_API_KEY", "")
    env["DF_API_URL"] = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
    subprocess.run(cmd, check=True, env=env)

# ============================================================
# Stage 1: Prepare (Strict & Robust)
# ============================================================
async def safe_prepare(paper_dir: Path, style: str, sem: asyncio.Semaphore):
    async with sem:
        pdf_path = paper_dir / "paper.pdf"
        prompt_path = paper_dir / "prompt.json"
        
        # [Checkpoint] 检查 Prompt 是否已生成且有效
        if is_file_valid(prompt_path, min_size=50):
            try:
                # 再次确认 JSON 格式，防止上次写了一半断电
                load_prompt_text(prompt_path)
                return 
            except:
                print(f"⚠️ [Fix] Prompt JSON corrupt, regenerating: {paper_dir.name}")

        if not pdf_path.exists(): 
            return

        print(f"[PREPARE] {paper_dir.name}")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _prepare_sync, paper_dir, pdf_path, style)
        except Exception as e:
            print(f"❌ Prepare Failed {paper_dir.name}: {e}")

def _prepare_sync(paper_dir, pdf_path, style):
    # 先写到 temp，再 rename，保证原子性（防止写一半挂了）
    text = extract_pdf_text(str(pdf_path))
    idea = run_paper_idea_extractor(text)
    desc = generate_fig_desc(idea, style)
    
    temp_path = paper_dir / "prompt.json.tmp"
    final_path = paper_dir / "prompt.json"
    
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"prompt": desc}, f, ensure_ascii=False, indent=2)
    
    os.replace(temp_path, final_path)

# ============================================================
# Stage 2: Background Gen (API + Graphviz)
# ============================================================
async def task_background_gen(paper_dir: Path, sem: asyncio.Semaphore, allowed_models: Set[str]):
    async with sem:
        prompt_path = paper_dir / "prompt.json"
        if not is_file_valid(prompt_path): return # 依赖检查
        
        prompt = load_prompt_text(prompt_path)
        api_key = os.getenv("DF_API_KEY")
        api_url = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")

        tasks = []

        # 1. Graphviz
        if "graphviz" in allowed_models:
            gv_out = paper_dir / "graphviz.png"
            if not is_file_valid(gv_out): # [Checkpoint]
                try:
                    await GZ.main(prompt, str(gv_out.with_suffix("")))
                except: pass

        # 2. API Models
        api_mapping = [
            ("gemini2.5", "gemini-2.5-flash-image-preview"), 
            ("dalle",     "dall-e-3")
        ]

        for name, model_id in api_mapping:
            if name in allowed_models:
                out = paper_dir / f"{name}.png"
                if not is_file_valid(out): # [Checkpoint]
                    tasks.append(GD.generate_and_save_image(prompt, str(out), model_id, api_url, api_key))


        if "seedream" in allowed_models:
            sd_out = paper_dir / "seedream.png"
            if not is_file_valid(sd_out):
                pipe, dev = seedream.load_seedream() # uses DF_API_URL env var
                tasks.append(asyncio.to_thread(seedream.generate, pipe, dev, prompt, sd_out))

        # Qwen 处理 (API版)
        if "qwen" in allowed_models:
            qw_out = paper_dir / "qwen.png"
            if not is_file_valid(qw_out):
                pipe, dev = qwen.load_qwen() # uses DF_API_URL env var
                tasks.append(asyncio.to_thread(qwen.generate, pipe, dev, prompt, qw_out))

        if tasks:
            await asyncio.gather(*tasks)

# ============================================================
# Stage 3: Local GPU Cycle (Robust Batching)
# ============================================================
def process_local_model_cycle(model_name: str, paper_dirs: List[Path], autobench_script: Path):
    """
    自带断点检测的 GPU 批处理
    """
    # [Resume Logic] 找出所有 "有 Prompt" 且 "图片未生成/损坏" 的任务
    todo_papers = []
    for p in paper_dirs:
        prompt_path = p / "prompt.json"
        img_path = p / f"{model_name}.png"
        
        # 必须有 prompt 且 (没有图 或 图坏了)
        if is_file_valid(prompt_path) and not is_file_valid(img_path):
            todo_papers.append(p)

    # 1. 生成阶段
    if todo_papers:
        print(f"\n🚀 [GPU GEN] Loading {model_name} for {len(todo_papers)} papers...")
        pipe = None
        device = None
        try:
            # Load
            if model_name == "sdxl":     pipe = sdxl.load_sdxl()
            elif model_name == "pixart": pipe, device = pixart.load_pixart()
            elif model_name == "flux":   pipe = flux.load_flux()
            elif model_name == "flux2":  pipe = flux2.load_flux2()

            # Batch Run
            for i, p in enumerate(todo_papers):
                prompt = load_prompt_text(p / "prompt.json")
                out = str(p / f"{model_name}.png")
                print(f"   Generating {model_name}: {p.name} ({i+1}/{len(todo_papers)})")
                try:
                    # 调用生成
                    if model_name == "sdxl":     sdxl.generate(pipe=pipe, prompt=prompt, out_path=out)
                    elif model_name == "pixart": pixart.generate(pipe=pipe, device=device, prompt=prompt, out_path=out)
                    elif model_name == "flux":   flux.generate(pipe=pipe, prompt=prompt, out_path=out)
                    elif model_name == "flux2":  flux2.generate(pipe=pipe, prompt=prompt, out_path=out)
                except Exception as e:
                    print(f"   ❌ Gen Error {p.name}: {e}")
                    # 如果生成失败，确保不留下垃圾文件
                    if os.path.exists(out): os.remove(out)

        except Exception as e:
            print(f"❌ Critical Model Load Error: {e}")
        finally:
            print(f"[{model_name}] Unloading...")
            del pipe
            if device: del device
            gc.collect()
            torch.cuda.empty_cache()
    else:
        print(f"✅ [Skip] All {model_name} images already exist.")

    # 2. 评测阶段 (即使生成阶段跳过了，也要检查是否缺评测)
    run_batch_autobench(model_name, paper_dirs, autobench_script)

def run_batch_autobench(img_name_prefix: str, paper_dirs: List[Path], autobench_script: Path):
    target_tasks = []
    for p in paper_dirs:
        img_path = p / f"{img_name_prefix}.png"
        out_dir = p / f"{img_name_prefix}_autobench"
        
        # [Resume Logic] 有效图片 且 (无 Bench 结果 或 Bench 结果损坏)
        if is_file_valid(img_path) and not is_bench_valid(out_dir):
            target_tasks.append((img_path, out_dir))
            
    if not target_tasks:
        return

    print(f"\n📉 [GPU BENCH] {img_name_prefix}: {len(target_tasks)} tasks remaining...")
    
    for i, (img, out) in enumerate(target_tasks):
        out.mkdir(parents=True, exist_ok=True)
        # 强制覆盖旧结果
        cmd = ["python", str(autobench_script), "--img", str(img), "--out", str(out)]
        try:
            run_subprocess_sync(cmd)
        except Exception as e:
            print(f"   ❌ Bench Error {img.name}: {e}")

# ============================================================
# Main
# ============================================================
async def main_async(args):
    roots = [Path(r).resolve() for r in args.root]
    
    # 0. Organize (幂等操作，每次跑都安全)
    if not args.skip_organize:
        for r in roots: organize_raw_pdfs(r)

    autobench_script = Path(args.autobench_script).resolve()
    
    # 1. Collect Papers
    paper_dirs = []
    for r in roots:
        paper_dirs.extend(find_paper_dirs(r))
    print(f"[Info] Scanning {len(paper_dirs)} papers for missing checkpoints...")

    # 2. Filter Models
    all_known_models = {
        "sdxl", "pixart", "flux", "flux2", "qwen", "seedream",
        "graphviz", "gemini2.5", "delle"
    }
    target_models = set(args.models) if args.models else all_known_models
    print(f"[Targets] {sorted(list(target_models))}")

    # --- Phase 1: Prepare ---
    if not args.skip_prepare:
        print("\n=== Phase 1: Prepare Prompts (Resume) ===")
        sem_prep = asyncio.Semaphore(args.max_prepare)
        await asyncio.gather(*[safe_prepare(d, args.style, sem_prep) for d in paper_dirs])

    # --- Phase 2: Background Gen ---
    bg_candidates = {"graphviz", "gemini2.5", "delle", "seedream", "qwen"}
    bg_targets = target_models.intersection(bg_candidates)
    
    bg_future = None
    if bg_targets:
        print(f"\n=== Phase 2: Background Gen {bg_targets} (Resume) ===")
        sem_bg = asyncio.Semaphore(5)
        bg_tasks = [task_background_gen(d, sem_bg, bg_targets) for d in paper_dirs]
        bg_future = asyncio.gather(*bg_tasks)

    # --- Phase 3: Local GPU ---
    local_candidates = ["sdxl", "pixart", "flux", "flux2"]
    local_targets = [m for m in local_candidates if m in target_models]

    if local_targets:
        for model in local_targets:
            # 自动跳过已完成的
            process_local_model_cycle(model, paper_dirs, autobench_script)

    # --- Phase 4: Finalize ---
    if bg_future:
        print("\n=== Phase 4: Waiting Background Tasks ===")
        await bg_future 
        for model in bg_targets:
            run_batch_autobench(model, paper_dirs, autobench_script)

    print("\n✅ All Pipeline Finished.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs='+', required=True)
    p.add_argument("--autobench_script", required=True)
    p.add_argument("--style", default="academic method diagram, clean white background")
    p.add_argument("--max_prepare", type=int, default=5)
    p.add_argument("--models", nargs="+")
    p.add_argument("--skip_organize", action="store_true")
    p.add_argument("--skip_prepare", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
