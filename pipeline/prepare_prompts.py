import os
import json
import argparse
from typing import List

import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from utils.utils import robust_parse_json

# ============================================================
# Step 0. PDF → text（严格等价 workflow）
# ============================================================
def extract_pdf_text(pdf_path: str, max_pages: int = 10) -> str:
    doc = fitz.open(pdf_path)
    texts: List[str] = []

    for i in range(min(len(doc), max_pages)):
        page = doc.load_page(i)
        texts.append(page.get_text("text") or "")

    return "\n".join(texts).strip()


# # ============================================================
# # 通用 JSON 解析（等价 robust_parse_json 的最小实现）
# # ============================================================
# def safe_json_loads(text: str) -> dict:
#     text = text.strip()

#     # 去 code fence
#     if text.startswith("```"):
#         text = text.split("```", 2)[1]

#     return json.loads(text)


# ============================================================
# Step 1. paper_idea_extractor（prompt 用你的，执行方式对齐）
# ============================================================
def run_paper_idea_extractor(paper_text: str) -> str:
    """
    PDF text → paper_idea
    """

    SYSTEM_PROMPT = """
    你现在的任务是：从提供的论文内容中，**精确抽取整篇论文的 “Methods”（方法）部分原文**。

    请严格遵守以下要求：

    1. **只做抽取，不做加工**  
      - 不要进行任何形式的解释、总结、改写或补充。  
      - 不要添加任何你自己的文字、标点或说明。  
      - 只返回从论文中截取出来的原始内容。

    2. **必须完整抽取 “Methods” 部分**  
      - 如果论文中有明确的章节标题，如 “Methods”, “Materials and Methods”, “Methodology” 等，请从该章节标题开始，到该章节正式结束为止，**原样抽取全部内容**。  
      - 如果论文中没有明确命名为 “Methods” 的章节，请抽取所有清晰描述研究方法、实验流程、算法、模型、技术方案等的内容。

    3. **保留原有结构与排版格式**  
      - 保留原来的段落分行、标题层级、列表、公式标记等文本结构。  
      - 不要擅自合并或拆分段落，不要改变任何文字顺序。

    4. **字符与内容要求**  
      - 不要引入新的控制字符或特殊符号。  
      - 尽量去除或避免返回 ASCII 控制字符（例如不可见的换页符、奇怪的转义符等），只保留正常可见文本。  
      - 不要在内容前后额外添加注释、标签或说明文字。

    5. **输出格式（必须是合法 JSON）**  
      - 最终回答必须是一个合法 JSON 对象，键为 `"paper_idea"`。  
      - JSON 字符串中不要出现未转义的换行控制字符或非法字符，避免 JSON 解析错误。  
      - 内容格式如下（注意是 JSON 而不是自然语言说明）：
      
    ```json
    {
      "paper_idea": "Paper title: xxx. Paper sections: original text of specific sections of paper...."
    }
"""


    TASK_PROMPT = """
Based on the paper content provided below, extract the **entire content of the Methods section**, ensuring that the structure and formatting of the original text are preserved. Do **not** summarize or interpret any part of the section. Return the content exactly as it appears.

**Important:**
1. Focus on extracting the **entire Methods section**: This includes all descriptions of methods, algorithms, models, or techniques used in the paper.
2. Preserve the **exact structure** and **formatting** of the original content.
3. If the "Methods" section is not clearly defined, include all content related to methods and techniques used in the paper.
4. 去掉多余移除 ASCII 控制字符，尽量以纯文本，形式返回，不要有多余符号，以免json解析错误！！！

Paper content: {paper_content}
"""
    # === 等价 parser.get_format_instruction() ===
    FORMAT_INSTRUCTION = """
请确保你的最终输出是**严格合法的 JSON 对象**，不要包含任何额外解释或文本。
"""


    llm = ChatOpenAI(
        openai_api_base=os.getenv("DF_API_URL","http://123.129.219.111:3000/v1"),
        openai_api_key=os.getenv("DF_API_KEY"),
        model_name="claude-haiku-4-5-20251001",
        temperature=0.0,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\n" + FORMAT_INSTRUCTION),
        HumanMessage(content=TASK_PROMPT.format(paper_content=paper_text)),
    ]

    resp = llm.invoke(messages)
    data = robust_parse_json(resp.content)

    return data["paper_idea"]


# ============================================================
# Step 2. FigureDescPrompts → fig_desc（prompt 用你的）
# ============================================================
def generate_fig_desc(paper_idea: str, style: str) -> str:
    """
    paper_idea → fig_desc
    """

    SYSTEM_PROMPT = """
你是一位世界顶级的 CVPR/NeurIPS 视觉架构师
你的核心能力是将晦涩难懂的论文逻辑，转化为**具体的、画面感极强的视觉描述。
"""

    TASK_PROMPT = """
下面是一篇论文的核心研究内容（paper_idea）：

{paper_idea}

请根据上述内容，编写一个用于 Text-to-Image 模型的英文提示词（Prompt）。

### 提示词编写策略：
1. 强调 “科研绘图”，用于论文插图；
2. **风格（Style）**：{style}
   - 必须强制包含论文内容的关键词.
3. 白色背景，然后要分成多个panel，就跟论文中的图一样，每个panel都要有自己的标题，标题要放在panel的上方；
4. 信息量要丰富，填满整个画面；

### 最终生成的 fig_desc 必须是一段连贯的英文描述;

# Output Format (The Golden Schema)
请严格遵守以下 JSON 输出要求：

1. 最终响应必须是一个严格合法的 JSON 对象，不能包含任何额外文字、解释或 Markdown 标记。
2. 该 JSON 对象只能包含一个键：fig_desc。
3. fig_desc 的值必须是一个字符串，用于描述整张图的视觉结构和内容。
4. 在 JSON 中：
   - 所有双引号必须写成 \\"；
   - 所有换行必须写成 \\n（不能直接换行输出）；
   - 不要包含制表符或其它控制字符。

示例（仅示意结构，实际内容请根据论文生成）：
{{
  "fig_desc": "xxx"
}}
"""
    FORMAT_INSTRUCTION = """
请确保你的最终输出是**严格合法的 JSON 对象**，
不要包含任何额外解释或 Markdown。
"""

    llm = ChatOpenAI(
        openai_api_base=os.getenv("DF_API_URL","http://123.129.219.111:3000/v1"),
        openai_api_key=os.getenv("DF_API_KEY"),
        model_name="claude-haiku-4-5-20251001",
        temperature=0.0,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\n" + FORMAT_INSTRUCTION),
        HumanMessage(content=TASK_PROMPT.format(
            paper_idea=paper_idea,
            style=style,
        )),
    ]

    resp = llm.invoke(messages)
    data = robust_parse_json(resp.content)

    return data["fig_desc"]

def save_prompt_json(fig_desc: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prompt.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"prompt": fig_desc},
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser("PDF → fig_desc (source-aligned)")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--style",
        default="academic method diagram, clean white background"
    )
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    
    print("[1] Reading PDF...")
    paper_text = extract_pdf_text(args.pdf)

    print("[2] Running paper_idea_extractor...")
    paper_idea = run_paper_idea_extractor(paper_text)

    print("[3] Running figure_desc_generator...")
    fig_desc = generate_fig_desc(paper_idea, args.style)

    prompt_json_path = os.path.join(out_dir, "prompt.json")
    with open(prompt_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"prompt": fig_desc},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ Done. prompt.json saved to: {prompt_json_path}")


if __name__ == "__main__":
    main()
