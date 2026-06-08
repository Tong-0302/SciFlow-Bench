import argparse
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import re
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from graphviz import Source

# ===========================================================
# 环境配置
# ===========================================================
API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
API_KEY = os.getenv("DF_API_KEY")
MODEL = os.getenv("DF_MODEL", "gpt-4o")


# ===========================================================
# LLM 调用逻辑
# ===========================================================
async def call_llm_for_graphviz_code(user_prompt: str,
                                     model: str = MODEL,
                                     api_url: str = API_URL,
                                     api_key: str = API_KEY) -> Optional[str]:
    """
    调用 LLM，将自然语言描述转换为 Graphviz DOT 代码
    """
    system_prompt = f"""
You are an expert Graphviz DOT code generator.
Your task is to convert the user's natural language description into valid Graphviz DOT syntax.

Rules:
1. Output ONLY the code inside a markdown code block: ```dot ... ``` 
2. No explanations.
3. Use meaningful node labels.
4. Use rankdir=LR unless user specifies otherwise.
5. Ensure correct DOT syntax.

User Description:
---
{user_prompt}
---
"""

    llm = ChatOpenAI(
        openai_api_base=api_url,
        openai_api_key=api_key,
        model_name=model,
        temperature=0.0,
    )

    response = await llm.ainvoke([HumanMessage(content=system_prompt)])
    content = response.content.strip()

    try:
        if "```dot" in content:
            return content.split("```dot")[1].split("```")[0].strip()
        elif "```graphviz" in content:
            return content.split("```graphviz")[1].split("```")[0].strip()
        elif "digraph" in content or "graph" in content:
            return content
        else:
            print("⚠️ Warning: LLM output may not be valid DOT code.")
            return content
    except Exception as e:
        print(f"❌ DOT 解析失败: {e}")
        return None


# ===========================================================
# 渲染逻辑（仅输出 PNG，不保存任何中间文件）
# ===========================================================
def render_dot_to_image(dot_code: str, output_filename: str):
    """
    使用 Graphviz 渲染 PNG，并删除渲染过程中生成的 .dot 文件
    """
    try:
        s = Source(dot_code, filename=output_filename, format="png")
        # cleanup=True 会自动删除中间的 .dot 文件
        output_path = s.render(view=False, cleanup=True)
        return output_path
    except Exception as e:
        print(f"❌ Graphviz 渲染失败: {e}")
        print("请检查是否已安装 Graphviz (sudo apt install graphviz)")
        return None


# ===========================================================
# 主流程
# ===========================================================
async def main(prompt: str, output_name: str):

    dot_code = await call_llm_for_graphviz_code(prompt)

    if not dot_code:
        print("⚠️ 未生成有效 DOT 代码")
        return

    print("🎨 正在渲染 PNG 图片...")

    final_path = render_dot_to_image(dot_code, output_name)

    if final_path:
        print(f"✅ 完成！输出图片路径: {final_path}")


# ===========================================================
# CLI
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM → Graphviz PNG")
    parser.add_argument("--prompt", required=True, help="Diagram description")
    parser.add_argument("--out", default="baseline_result", help="Output PNG filename")
    args = parser.parse_args()

    asyncio.run(main(args.prompt, args.out))
