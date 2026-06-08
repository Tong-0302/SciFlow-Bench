import os
import json
import argparse
import asyncio

# ===== import eval modules =====
from evaluation import eval_graph_level as graph_level
from evaluation import eval_text_level as text_level
from evaluation import eval_image_level as image_level


# ============================================================
# Config
# ============================================================

BASELINES = [
    # "graphviz",
    # "sdxl",
    # "pixart",
    # "flux",
    # "qwen",
    # "gemini2.5",
    # "dalle",
    "seedream"
]

# BASELINES = [
#     "gemini3"
# ]
GRAPH_WEIGHT = 0.4
TEXT_WEIGHT  = 0.3
IMAGE_WEIGHT = 0.3

MAX_CONCURRENT_TASKS = 5


# ============================================================
# Resume Utils
# ============================================================

def load_existing_results(path: str):
    """
    Load existing evaluation results for resume.
    Returns:
        results: list
        done_set: set of (paper_id, baseline)
    """
    if not path or not os.path.exists(path):
        return [], set()

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    done = set()
    for r in results:
        done.add((r["paper_id"], r["baseline"]))

    print(f"[Resume] Loaded {len(done)} completed (paper, baseline) pairs")
    return results, done


# ============================================================
# Utils
# ============================================================

class EvalFailure(Exception):
    def __init__(self, paper_id, baseline, stage, error):
        super().__init__(str(error))
        self.paper_id = paper_id
        self.baseline = baseline
        self.stage = stage
        self.error = str(error)


def find_graph_with_fallback(dir_path: str) -> str:
    for name in ["canvas_full.json", "canvas_full_merge.json", "canvas_full_step10.json"]:
        p = os.path.join(dir_path, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No canvas_full found in {dir_path}")


def find_gt_image(paper_dir: str) -> str:
    candidates = [
        os.path.join(paper_dir, "sam_results", "segmentation_result.png"),
        os.path.join(paper_dir, "segmentation_result.png"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"GT image not found. Tried: {candidates}"
    )


def find_pred_image(paper_dir: str, baseline: str) -> str:
    p = os.path.join(paper_dir, f"{baseline}.png")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"No predicted image for {baseline}")


# ============================================================
# Core Eval (single baseline)
# ============================================================

async def eval_one(
    paper_dir: str,
    paper_id: str,
    baseline: str,
    visual_metrics_engine: image_level.VisualMetrics,
    semaphore: asyncio.Semaphore
):
    async with semaphore:
        try:
            prompt_json = os.path.join(paper_dir, "prompt.json")
            if not os.path.exists(prompt_json):
                raise FileNotFoundError("Missing prompt.json")

            gt_graph = find_graph_with_fallback(paper_dir)
            pred_graph_dir = os.path.join(paper_dir, f"{baseline}_autobench")
            pred_graph = find_graph_with_fallback(pred_graph_dir)

            gt_image   = find_gt_image(paper_dir)
            pred_image = find_pred_image(paper_dir, baseline)

            # Graph-Level
            graph_res = await asyncio.to_thread(
                graph_level.evaluate, gt_graph, pred_graph
            )
            graph_score = float(graph_res["final_score"])

            # Text-Level
            text_res = await text_level.evaluate(
                method_path=prompt_json,
                graph_path=pred_graph
            )
            text_score = float(text_res["metrics"]["final_text_score"])

            # Image-Level
            image_res = await image_level.evaluate(
                img_path=pred_image,
                gt_img_path=gt_image,
                method_path=prompt_json,
                graph_json_path=pred_graph,
                metrics_instance=visual_metrics_engine
            )
            image_score = float(image_res["metrics"]["final_visual_score"])

            overall = (
                GRAPH_WEIGHT * graph_score +
                TEXT_WEIGHT  * text_score +
                IMAGE_WEIGHT * image_score
            )

            return {
                "paper_id": paper_id,
                "baseline": baseline,
                "graph_level": graph_res,
                "text_level":  text_res["metrics"],
                "image_level": image_res["metrics"],
                "scores": {
                    "graph": graph_score,
                    "text":  text_score,
                    "image": image_score,
                    "overall": overall
                }
            }

        except Exception as e:
            raise EvalFailure(paper_id, baseline, "eval", e)


# ============================================================
# Main Loop (Resume-enabled)
# ============================================================

async def run_evaluation(root: str, out_path: str, fail_log_path: str):
    paper_dirs = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]

    print(f"[Info] Found {len(paper_dirs)} paper dirs")

    # -------- resume --------
    results, done_set = load_existing_results(out_path)
    fail_logs = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    print("[Init] Loading VisualMetrics once...")
    visual_metrics_engine = image_level.VisualMetrics()

    for paper_dir in paper_dirs:
        paper_id = os.path.basename(paper_dir)
        print(f"\n📄 {paper_id}")

        tasks = []
        active_baselines = []

        for baseline in BASELINES:
            key = (paper_id, baseline)
            if key in done_set:
                print(f"[Resume] Skip {paper_id} | {baseline}")
                continue

            tasks.append(
                eval_one(
                    paper_dir,
                    paper_id,
                    baseline,
                    visual_metrics_engine,
                    semaphore
                )
            )
            active_baselines.append(baseline)

        if not tasks:
            continue

        paper_results = await asyncio.gather(*tasks, return_exceptions=True)

        for baseline, res in zip(active_baselines, paper_results):
            if isinstance(res, EvalFailure):
                print(f"[Fail] {res.paper_id} | {res.baseline} | {res.error}")
                fail_logs.append({
                    "paper_id": res.paper_id,
                    "baseline": res.baseline,
                    "stage": res.stage,
                    "error": res.error
                })
                continue

            results.append(res)
            done_set.add((res["paper_id"], res["baseline"]))

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if fail_log_path:
                os.makedirs(os.path.dirname(fail_log_path), exist_ok=True)
                
            # ---- critical: flush immediately ----
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    if fail_log_path:
        with open(fail_log_path, "w", encoding="utf-8") as f:
            json.dump(fail_logs, f, indent=2, ensure_ascii=False)

    print("\n✅ Evaluation completed.")
    print(f"📄 Results saved to: {out_path}")
    print(f"📊 Success: {len(results)} | Failed: {len(fail_logs)}")


# ============================================================
# Entry
# ============================================================

def main():
    print(f"DEBUG: HF_HOME is set to: {os.environ.get('HF_HOME')}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fail-log", default=None)
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.root, args.out, args.fail_log))


if __name__ == "__main__":
    main()



# python -m dataflow_agent.bench.baseline_eval --root autobench/cv_2025_autobench --out aaresults/cv_2025_autobench/baseline_eval_results.json --fail-log aaresults/cv_2025_autobench/failures.json

