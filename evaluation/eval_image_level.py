import argparse
import json
import os
import asyncio
import re
from types import SimpleNamespace
from typing import Dict, List, Any

import torch
import torchvision.io as io
import torchvision.transforms.functional as F
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller

from dotenv import load_dotenv
load_dotenv() 

# ============================================================
# 1. 图像处理 Utils (移除 Base64 编码函数)
# ============================================================
def load_graph_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prompt"]

def extract_visual_elements_from_json(graph_json: Dict) -> str:
    """
    Extract lightweight visual entity hints from Graph JSON.
    Note: This function intentionally ignores high-level chunks.
    """
    nodes = []

    for n in graph_json.get("nodes", []):
        text = (
            n.get("desc", "").strip()
            or n.get("vlm_prompt", "").strip()
            or n.get("node_type", "")
        )
        if text:
            nodes.append(text)

    if not nodes:
        return ""

    uniq_nodes = []
    for t in nodes:
        if t not in uniq_nodes:
            uniq_nodes.append(t)

    return f"Detected Components: {', '.join(uniq_nodes[:30])}"


CLIP_MAX = 100.0  

# # ============================================================
# # 2. PyTorch Metrics Class (保持不变)
# # ============================================================
# class VisualMetrics:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Loading visual metrics on {self.device} (This should only happen once)...")
        
#         self.clip_metric = CLIPScore(
#             model_name_or_path="openai/clip-vit-base-patch32"
#         ).to(self.device)
#         self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
#             net_type='alex', normalize=True
#         ).to(self.device)
#         self.clip_metric.eval()
#         self.lpips_metric.eval()

#     def compute_clip(self, img_path: str, text: str):
#         try:
#             image = io.read_image(img_path).float() / 255.0
#             image = image.unsqueeze(0).to(self.device)
#             raw_score = self.clip_metric(image, text)
#             raw_score = float(raw_score.detach().cpu())

#             clip_norm = raw_score / CLIP_MAX
#             clip_norm = min(1.0, max(0.0, clip_norm))

#             return clip_norm, raw_score
#         except Exception as e:
#             print(f"[Metric Error] CLIP failed: {e}")
#             return 0.0, 0.0

#     def compute_lpips(self, pred_path: str, gt_path: str):
#         try:
#             img_pred = io.read_image(pred_path).to(self.device).float() / 255.0
#             img_gt = io.read_image(gt_path).to(self.device).float() / 255.0
#             img_pred = img_pred.unsqueeze(0)
#             img_gt = img_gt.unsqueeze(0)
#             if img_pred.shape[-2:] != img_gt.shape[-2:]:
#                 img_pred = F.resize(img_pred, img_gt.shape[-2:])
#             d = self.lpips_metric(img_pred, img_gt)
#             sim = 1.0 - float(d.detach().cpu())
#             sim = min(1.0, max(0.0, sim))
#             return sim
#         except Exception as e:
#             print(f"[Metric Error] LPIPS failed: {e}")
#             return 0.0
# ============================================================
# 2. PyTorch Metrics Class (修复版)
# ============================================================
# class VisualMetrics:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Loading visual metrics on {self.device} (This should only happen once)...")
        
#         # CLIP: 推荐输入 uint8
#         self.clip_metric = CLIPScore(
#             model_name_or_path="openai/clip-vit-base-patch32"
#         ).to(self.device)
        
#         # LPIPS: 需要 normalize=True，期望 float [0,1]
#         self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
#             net_type='alex', normalize=True
#         ).to(self.device)
        
#         self.clip_metric.eval()
#         self.lpips_metric.eval()

#     def compute_clip(self, img_path: str, text: str):
#         try:
#             # === CLIP 修复 ===
#             # 1. 读取原始 uint8 数据 [C, H, W]
#             image = io.read_image(img_path) 
            
#             # 2. 确保是 RGB (3通道)，去掉 Alpha 通道
#             if image.shape[0] == 4:
#                 image = image[:3]
#             elif image.shape[0] == 1:
#                 image = image.repeat(3, 1, 1)

#             # 3. 保持 uint8 类型 (0-255)，不要除以 255.0
#             # CLIPScore 内部的 Preprocessor 更喜欢 uint8
#             image = image.to(dtype=torch.uint8)
            
#             # 4. 增加 Batch 维度 [1, C, H, W]
#             image = image.unsqueeze(0).to(self.device)

#             # 计算
#             raw_score = self.clip_metric(image, text)
#             raw_score = float(raw_score.detach().cpu())

#             # 归一化 (CLIP 分数通常在 20-30 左右，这里粗略归一化)
#             clip_norm = raw_score / CLIP_MAX
#             clip_norm = min(1.0, max(0.0, clip_norm))

#             return clip_norm, raw_score
#         except Exception as e:
#             print(f"[Metric Error] CLIP failed: {e}")
#             return 0.0, 0.0

#     def compute_lpips(self, pred_path: str, gt_path: str):
#         try:
#             # === LPIPS 修复 ===
#             # 1. 读取并转 float [0, 1]
#             img_pred = io.read_image(pred_path).float() / 255.0
#             img_gt   = io.read_image(gt_path).float() / 255.0

#             # 2. 强制取前3通道 (RGB)，防止 PNG 的 Alpha 通道报错
#             if img_pred.shape[0] > 3: img_pred = img_pred[:3]
#             if img_gt.shape[0] > 3:   img_gt = img_gt[:3]

#             # 3. !!! 关键修复 !!! 强制截断到 [0, 1]
#             # 浮点运算可能会产生 1.0000001 导致 LPIPS 报错
#             img_pred = torch.clamp(img_pred, 0.0, 1.0)
#             img_gt   = torch.clamp(img_gt, 0.0, 1.0)

#             # 4. 统一 Resize 到 256x256
#             TARGET_SIZE = (256, 256)
#             img_pred = F.resize(img_pred, TARGET_SIZE)
#             img_gt   = F.resize(img_gt, TARGET_SIZE)

#             # 5. 增加 Batch 维度
#             img_pred = img_pred.unsqueeze(0).to(self.device)
#             img_gt   = img_gt.unsqueeze(0).to(self.device)

#             # 计算
#             d = self.lpips_metric(img_pred, img_gt)
#             sim = 1.0 - float(d.detach().cpu())
#             return min(1.0, max(0.0, sim))

#         except Exception as e:
#             print(f"[Metric Error] LPIPS failed: {e}")
#             return 0.0
class VisualMetrics:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading visual metrics on {self.device}...")
        
        self.clip_metric = CLIPScore(
            model_name_or_path="openai/clip-vit-base-patch32"
        ).to(self.device)
        
        # =====================================================
        # 修改点 1: normalize=False
        # 告诉 torchmetrics 不要自己做归一化检查，我们手动做
        # =====================================================
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', 
            normalize=False 
        ).to(self.device)
        
        self.clip_metric.eval()
        self.lpips_metric.eval()

    def compute_clip(self, img_path: str, text: str):
        try:
            image = io.read_image(img_path)
            if image.shape[0] == 4: image = image[:3]
            elif image.shape[0] == 1: image = image.repeat(3, 1, 1)
            
            image = image.to(dtype=torch.uint8)
            image = image.unsqueeze(0).to(self.device)
            
            raw_score = self.clip_metric(image, text)
            raw_score = float(raw_score.detach().cpu())
            
            clip_norm = raw_score / CLIP_MAX
            clip_norm = min(1.0, max(0.0, clip_norm))
            return clip_norm, raw_score
        except Exception as e:
            print(f"[Metric Error] CLIP failed: {e}")
            return 0.0, 0.0

    def compute_lpips(self, pred_path: str, gt_path: str):
        try:
            # 1. 读取并转 float [0, 1]
            img_pred = io.read_image(pred_path).float() / 255.0
            img_gt   = io.read_image(gt_path).float() / 255.0

            # 2. 确保 RGB
            if img_pred.shape[0] > 3: img_pred = img_pred[:3]
            if img_gt.shape[0] > 3:   img_gt = img_gt[:3]

            # 3. 截断到 [0, 1] (消除 1.000001 这种误差)
            img_pred = torch.clamp(img_pred, 0.0, 1.0)
            img_gt   = torch.clamp(img_gt, 0.0, 1.0)

            # 4. Resize
            TARGET_SIZE = (256, 256)
            img_pred = F.resize(img_pred, TARGET_SIZE)
            img_gt   = F.resize(img_gt, TARGET_SIZE)

            # =====================================================
            # 修改点 2: 手动归一化到 [-1, 1]
            # 公式: x * 2 - 1
            # =====================================================
            img_pred = img_pred * 2.0 - 1.0
            img_gt   = img_gt * 2.0 - 1.0

            # 再次截断，确保数值绝对在 [-1, 1] 之间
            img_pred = torch.clamp(img_pred, -1.0, 1.0)
            img_gt   = torch.clamp(img_gt, -1.0, 1.0)

            # 5. 增加 Batch 维度
            img_pred = img_pred.unsqueeze(0).to(self.device)
            img_gt   = img_gt.unsqueeze(0).to(self.device)

            # 计算
            d = self.lpips_metric(img_pred, img_gt)
            sim = 1.0 - float(d.detach().cpu())
            return min(1.0, max(0.0, sim))

        except Exception as e:
            # 打印更详细的错误堆栈，万一还有错方便查
            import traceback
            print(f"[Metric Error] LPIPS failed: {e}")
            traceback.print_exc()
            return 0.0
# ============================================================
# 3. VLM Judge
# # ============================================================
# async def vlm_visual_flow_judge(state, img_path: str, method_text: str, visual_elements_str: str):
#     caller = VisionLLMCaller(
#         state=state,
#         vlm_config={"mode": "understanding", "input_image": img_path}
#     )

#     prompt_text = f"""
# You are an expert scientific figure reviewer.
# Your task is to assess the **Visual Flow Consistency** of a generated diagram based on its method description.

# [Method Description]
# {method_text}

# [Key Visual Elements Detected in Diagram]
# (The following list summarizes textual labels detected from the diagram for reference only.)
# {visual_elements_str}

# [Evaluation Objective]
# Look at the input image. Evaluate whether the **visual organization** is logically consistent with the process described in the text.

# Focus on:
# 1. Directionality: Does the diagram flow (e.g., arrows, layout) follow the logical steps described in the text?
# 2. Spatial Logic: Are the major components arranged in a visually reasonable order (e.g., Left-to-Right or Top-to-Bottom)?

# [Scoring]
# Return a score between **0.0 and 1.0**:
# - 1.0 = Visual flow perfectly matches logic.
# - 0.0 = Visually chaotic or misleading.

# Output STRICT JSON ONLY:
# {{
#   "flow_score": 0.85,
#   "reason": "The diagram correctly follows a left-to-right pipeline structure matching the encoder-decoder logic."
# }}
# """

#     msg = await caller.call([HumanMessage(content=prompt_text)])
#     raw = msg.content.strip()

#     try:
#         j = json.loads(raw)
#         return float(j.get("flow_score", 0.0)), j.get("reason", "")
#     except:
#         m = re.search(r"\{.*\}", raw, re.S)
#         if m:
#             try:
#                 j = json.loads(m.group(0))
#                 return float(j.get("flow_score", 0.0)), j.get("reason", "")
#             except:
#                 pass
    
#     return 0.0, "JSON Parsing Failed"

async def vlm_visual_flow_judge(state, img_path: str, method_text: str, visual_elements_str: str, compress_quality: int = 75):
    """
    Args:
        compress_quality (int): JPEG 压缩质量 (1-100)，默认 75。
    """
    temp_img_path = None
    
    try:
        # ==================== 1. 图像预处理与压缩 ====================
        with Image.open(img_path) as img:
            # 如果是 RGBA (带透明度的 PNG) 或 P 模式，转换为 RGB，否则保存为 JPEG 会报错
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # 创建一个命名的临时文件，delete=False 允许我们在关闭文件后通过路径访问它
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_img_path = tmp_file.name
                # 保存压缩后的图像
                img.save(temp_img_path, format="JPEG", quality=compress_quality)
        
        # ==================== 2. 调用 VLM (使用临时路径) ====================
        # 注意：这里传入的是 temp_img_path 而不是原始的 img_path
        caller = VisionLLMCaller(
            state=state,
            vlm_config={"mode": "understanding", "input_image": temp_img_path}
        )

        prompt_text = f"""
You are an expert scientific figure reviewer.
Your task is to assess the **Visual Flow Consistency** of a generated diagram based on its method description.

[Method Description]
{method_text}

[Key Visual Elements Detected in Diagram]
(The following list summarizes textual labels detected from the diagram for reference only.)
{visual_elements_str}

[Evaluation Objective]
Look at the input image. Evaluate whether the **visual organization** is logically consistent with the process described in the text.

Focus on:
1. Directionality: Does the diagram flow (e.g., arrows, layout) follow the logical steps described in the text?
2. Spatial Logic: Are the major components arranged in a visually reasonable order (e.g., Left-to-Right or Top-to-Bottom)?

[Scoring]
Return a score between **0.0 and 1.0**:
- 1.0 = Visual flow perfectly matches logic.
- 0.0 = Visually chaotic or misleading.

Output STRICT JSON ONLY:
{{
  "flow_score": 0.85,
  "reason": "The diagram correctly follows a left-to-right pipeline structure matching the encoder-decoder logic."
}}
"""

        msg = await caller.call([HumanMessage(content=prompt_text)])
        raw = msg.content.strip()

        # ==================== 3. 解析结果 ====================
        try:
            j = json.loads(raw)
            return float(j.get("flow_score", 0.0)), j.get("reason", "")
        except:
            # 尝试使用正则提取 JSON (应对 VLM 输出包含 markdown 代码块的情况)
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    j = json.loads(m.group(0))
                    return float(j.get("flow_score", 0.0)), j.get("reason", "")
                except:
                    pass
        
        return 0.0, "JSON Parsing Failed"

    except Exception as e:
        return 0.0, f"Error during processing: {str(e)}"

    finally:
        # ==================== 4. 清理临时文件 ====================
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except OSError:
                pass
# ============================================================
# Main Pipeline
# ============================================================
async def evaluate(img_path, gt_img_path, method_path, graph_json_path, metrics_instance: VisualMetrics):
    print(f"\n[Evaluating Image] {img_path}")
    
    method_text = load_prompt_text(method_path)
    graph_json = load_graph_json(graph_json_path)
    visual_elements = extract_visual_elements_from_json(graph_json)

    clip_norm, clip_raw = metrics_instance.compute_clip(img_path, method_text)
    lpips_score = metrics_instance.compute_lpips(img_path, gt_img_path)

    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1"),
            api_key=os.getenv("DF_API_KEY"),
            model=os.getenv("DF_MODEL", "gpt-4o")
        )
    )

    flow_score, flow_reason = await vlm_visual_flow_judge(
        state, img_path, method_text, visual_elements
    )
    flow_score = min(1.0, max(0.0, flow_score))

    final_visual_score = (
        0.4 * clip_norm +
        0.2 * lpips_score +
        0.4 * flow_score
    )

    return {
        "metrics": {
            "clip_consistency": clip_norm,
            "clip_raw_score": clip_raw,
            "lpips_similarity": lpips_score,
            "visual_flow_score": flow_score,
            "final_visual_score": final_visual_score
        },
        "details": {
            "vlm_reason": flow_reason,
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--gt_img", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--graph", required=True)
    args = parser.parse_args()

    print("=== Initializing Global Models (Once) ===")
    global_metrics_engine = VisualMetrics()
    print("=== Initialization Complete. Starting Evaluation Loop ===")

    result = asyncio.run(evaluate(
        img_path=args.img,
        gt_img_path=args.gt_img,
        method_path=args.method,
        graph_json_path=args.graph,
        metrics_instance=global_metrics_engine
    ))
    
    print(json.dumps(result, indent=2, ensure_ascii=False))




# python -m dataflow_agent.bench.image_level --img autobench/cv_2025_autobench/2505.00752v2/gemini3.png --gt_img  autobench/cv_2025_autobench/2505.00752v2/sam_results/segmentation_result.png --method autobench/cv_2025_autobench/2505.00752v2/prompt.json --graph autobench/cv_2025_autobench/2505.00752v2/gemini3_autobench/canvas_full.json
