import argparse
import asyncio
import json
import os
import re
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ===========================================================
# Environment Configuration
# ===========================================================
API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1")
API_KEY = os.getenv("DF_API_KEY")
MODEL = os.getenv("DF_MODEL", "gpt-4o")


# ===========================================================
# LLM Calling Logic
# ===========================================================
async def call_llm_for_titles(md_text: str,
                               model: str = MODEL,
                               api_url: str = API_URL,
                               api_key: str = API_KEY) -> Optional[dict]:
    """
    Call LLM to identify and extract the method section title and figure title.
    """

    # Replace `{` and `}` with escaped characters to avoid f-string issues
    md_text_escaped = md_text.replace("{", "{{").replace("}", "}}")

    # Construct the prompt with the escaped text
    prompt = """
    You are a research text analysis assistant.

    Below is the OCR-extracted markdown content from a scientific paper.
    Please do the following:

    1. Read all figure captions thoroughly.
    2. Identify the **single figure** that describes the **internal architecture or structure** of the model. This figure typically contains detailed components like:
    - Encoders
    - Decoders
    - Feature aggregation modules
    - Feature fusion modules
    - Saliency or quality prediction branches
    - Shared encoders
    - Multi-branch structures
    - Any breakdown of modules inside the model.
    3. **Do not select figures** that describe workflow, pipeline, or end-to-end processes (e.g., sequences from input to output or the optimization process), as these do not represent internal architecture.
    4. Choose the figure that provides the **richest and most comprehensive description** of the model’s internal structure.
    5. Return only the **figure title** that best describes the model's internal architecture. The title should follow the format `Fig. X` (e.g., `Fig. 1` or `Fig. 9`).
    6. If no figure describes the internal architecture, return `"null"`.

    Markdown content:
    ---
    """ + md_text_escaped + """
    ---
    Please output:
    {
    "method_title": "Method Section Title (must start with '## ')",
    "figure_title": "Figure Title (must start with 'Fig.')"
    }
    """

    llm = ChatOpenAI(
        openai_api_base=api_url,
        openai_api_key=api_key,
        model_name=model,
        temperature=0.2,
    )

    message = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(message)
    content = response.content.strip()

    # Remove the surrounding backticks if present, then try parsing the JSON response
    content = content.strip('```json').strip('```')

    # Attempt to parse the response as JSON
    try:
        titles = json.loads(content)
    except Exception:
        # If JSON parsing fails, return the raw string and log it
        print(f"⚠️ JSON parsing failed, raw response: {content}")
        titles = content.strip()

    return titles


# ===========================================================
# Extract Content Based on Titles (with improved handling)
# ===========================================================
def extract_section_by_title(md_text: str, start_title: str) -> str:
    """
    Extract content from the markdown based on a given section title.
    提取从 start_title 开始，直到下一个非分页主标题（## 非第x页）为止的所有内容
    包含子标题（###）和分页标题（## 第x页）
    """
    # 正则逻辑：
    # 1. ^re.escape(start_title)\s*\n：匹配起始标题（整行），后面跟换行
    # 2. ([\s\S]*?)：非贪婪匹配所有内容（包括换行），直到触发终止条件
    # 3. (?=\n## (?!第 \d+ 页)|\Z)：终止条件（正向先行断言）：
    #    - 要么遇到 "\n## " 且后面不是「第 x 页」（即非分页主标题）
    #    - 要么到文档结束（\Z）
    pattern = re.compile(
        rf"^{re.escape(start_title)}\s*\n([\s\S]*?)(?=\n## (?!第 \d+ 页)|\Z)",
        re.MULTILINE
    )
    
    match = pattern.search(md_text)
    
    if match:
        section_content = match.group(1).strip()
        # 若需要保留内容开头的空行（如果原文本有），可去掉 strip()，直接返回 match.group(1)
        return section_content
    return ""


def extract_figure_description(md_text: str, figure_title: str) -> str:
    """
    精确提取：找到 <center> 标签，且该标签内部的内容必须以 figure_title 中的数字开头。
    兼容 "Fig. 9", "Figure 9", "Fig 9" 等多种前缀写法。
    不限制紧跟在数字后面的符号，提取直到 </center> 结束的所有内容。
    """
    if not figure_title:
        return ""

    match_num = re.search(r"(\d+)", figure_title)
    if not match_num:
        return figure_title 
    
    fig_num = match_num.group(1)
    
    pattern = re.compile(
        rf"<center>\s*((?:Fig(?:ure)?\.?)\s*{fig_num}(?!\d)[\s\S]*?)</center>", 
        re.IGNORECASE
    )

    match = pattern.search(md_text)

    if match:
        return match.group(1).strip()
    
    # 如果没找到严格匹配的，返回原标题作为兜底
    return figure_title


# ===========================================================
# CLI Mode for Testing
# ===========================================================
async def main(md_path: str, out_path: str):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"❌ File not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    print(f"🚀 Analyzing Markdown: {md_path}")

    # Get method section and figure title from LLM
    titles = await call_llm_for_titles(md_text)

    # Check if titles is a string, and handle accordingly
    if isinstance(titles, str):
        print(f"⚠️ Received raw response: {titles}")
        # If it's a string, manually extract method and figure titles
        # You can further enhance this by using regex or other logic
        method_title = "Unknown"
        figure_title = "Unknown"
    else:
        method_title = titles.get("method_title", "")
        figure_title = titles.get("figure_title", "")

    if not method_title or not figure_title:
        print("⚠️ Method section or figure title not detected.")
        return

    # Extract content for method section and figure description
    method_section = extract_section_by_title(md_text, method_title)
    figure_description = extract_figure_description(md_text, figure_title)

    print("✅ Method Section Title:\n", method_title)
    print("✅ Method Section Content:\n", method_section)
    print("✅ Figure Title:\n", figure_title)
    print("✅ Figure Description:\n", figure_description)

    # Output in JSONL format
    with open(out_path, "w", encoding="utf-8") as f:
        jsonl_data = {
            "method_title": method_title,
            "method_section": method_section,
            "figure_title": figure_title,
            "figure_description": figure_description
        }
        json.dump(jsonl_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {out_path}")


# ===========================================================
# CLI Interface
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract method title, method section, and figure description from OCR Markdown")
    parser.add_argument("--md", required=True, help="Path to OCR-extracted markdown file")
    parser.add_argument("--out", required=True, help="Output directory for the results")
    args = parser.parse_args()
    asyncio.run(main(args.md, args.out))

