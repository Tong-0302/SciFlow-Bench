import argparse
import asyncio
import json
import os
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ===========================================================
# 环境配置
# ===========================================================
API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
API_KEY = os.getenv("DF_API_KEY")
MODEL = os.getenv("DF_MODEL", "gpt-4o")


# ===========================================================
# LLM 调用逻辑
# ===========================================================
async def call_llm_for_framework_figure(md_text: str,
                                        model: str = MODEL,
                                        api_url: str = API_URL,
                                        api_key: str = API_KEY) -> Optional[str]:
    """
    调用 LLM，从 Markdown 中识别唯一框架图像文件名
    """
    prompt = f"""
You are a text analysis agent specialized in identifying the single figure that presents
the most detailed internal model architecture from OCR-extracted markdown.

Each figure appears in the format:
![](images/pageX_Y.jpg)
<center>Figure N: Caption text ...</center>

Your task:
1. Read all figure captions thoroughly.
2. Identify the single figure that describes the internal architecture or internal structure of the model.
3. Prioritize figures that include detailed component-level descriptions such as encoders, decoders, feature aggregation modules, feature fusion modules, saliency or quality prediction branches, shared encoders, multi-branch structures, or any breakdown of modules inside the model.
4. Do not select workflow, pipeline, or end-to-end process figures. These are diagrams that describe the sequence from input to output or the optimization process rather than the internal architecture.
5. Select the figure that provides the richest and most comprehensive description of the model’s internal structure.
6. Return only the image file name, formatted strictly as a JSON string.
7. If no figure describes internal architecture, return `"null"`.

Markdown content:
---
{md_text}
---
Output format:
"pageX_Y.jpg"  OR  "null"
"""

    llm = ChatOpenAI(
        openai_api_base=api_url,
        openai_api_key=api_key,
        model_name=model,
        temperature=0.0,
    )

    message = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(message)
    content = response.content.strip()

    # -------- JSON 解析 --------
    figure = None
    try:
        figure = json.loads(content)
        if isinstance(figure, list) and figure:  # 若模型误输出列表
            figure = figure[0]
    except Exception:
        import re
        match = re.search(r'page\d+_\d+\.jpg', content)
        if match:
            figure = match.group(0)

    return figure


# ===========================================================
# Workflow 调用入口
# ===========================================================
async def framework_figure_selector_entry(state):
    """
    DataFlow-Agent Workflow 节点入口
    """
    md_text = state.request.get("md_text", "")
    figure = await call_llm_for_framework_figure(md_text)
    state.agent_results["framework_figure_selector"] = {
        "results": {"framework_figure": figure}
    }
    print(f"✅ 框架图识别结果: {figure}")
    return state


# ===========================================================
# CLI 模式测试逻辑
# ===========================================================
async def main(md_path: str):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"❌ 未找到文件: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    print(f"🚀 正在分析 Markdown: {md_path}")
    figure = await call_llm_for_framework_figure(md_text)

    if not figure:
        print("⚠️ 未检测到框架图，请检查模型输出。")
    else:
        print(f"✅ 检测到框架图: {figure}")

    # 仅测试模式下输出文件（workflow 模式不写）
    out_path = os.path.splitext(md_path)[0] + "_framework_figure.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"framework_figure": figure}, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存到: {out_path}")


# ===========================================================
# CLI 接口
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect single framework figure from OCR Markdown")
    parser.add_argument("--md", required=True, help="Path to OCR-extracted markdown file")
    args = parser.parse_args()
    asyncio.run(main(args.md))
