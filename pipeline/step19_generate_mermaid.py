import os
import json
import argparse
import asyncio
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from utils.llm_callers.image import VisionLLMCaller
from utils.state import MainState

DATA_FLOW_PROMPT = """
You are given an image containing a scientific or system diagram.

Your task:
Extract all arrow-based dataflow connections and output ONLY Mermaid code.

============================================================
INTERNAL REQUIREMENTS (MUST FOLLOW)
============================================================

1. Identify every visually distinct node.
   - A node is any bounded region, block, box, icon, or element that participates in a connection.
   - The node’s visible text must be used as its semantic label.
   - If multiple nodes share the same visible text, they must receive unique instance names by adding an index suffix.
   - If a node has no visible text, assign a short neutral label derived only from its visual role.
   - Never merge visually separate nodes.
   - Never invent semantic content beyond what is visible.

2. Identify every arrow.
   - Detect all directional connectors regardless of shape or curvature.
   - Every arrow must map to exactly one start node and one end node.
   - Missing or skipped arrows is not allowed.
   - Ignore any text written directly on the connectors.

3. ABSOLUTE RESTRICTION: NO PATTERN COMPLETION
   - Do NOT assume any intermediate nodes even if the architecture looks repetitive.
   - Do NOT insert Conv, PixelShuffle, HFR, or any other node unless it is explicitly drawn.
   - A connection may directly go from PixelShuffle → HFR if the arrow touches those two boxes.
   - If the arrow visually connects only two boxes, output exactly those two.
   - NO inferred or “expected” layers. ZERO hallucination allowed.

4. Node naming.
   - Node name format: {label}_{index}
   - Label must be exactly the visible text inside the node.
   - Index assigned in spatial order.
   - Names must be unique across the entire diagram.

5. Output format.
   - Output ONLY Mermaid code:

        graph TD;
            A --> B;
            B --> C;

   - No explanation.
   - No JSON.
   - No node lists.
   - No comments.
   - Mermaid only.

============================================================
Now analyze the diagram and output ONLY the Mermaid graph.
"""


async def analyze_image_data_flow(state, img_path: str) -> str:
    caller = VisionLLMCaller(
        state=state,
        vlm_config={
            "mode": "understanding",
            "input_image": img_path,
            "timeout": 180,
        },
    )
    msg = await caller.call([HumanMessage(content=DATA_FLOW_PROMPT.strip())])
    return msg.content.strip()

async def process_image_and_generate_mermaid(imgs_dir: str, output_dir: str):
    from types import SimpleNamespace

    api_url = os.getenv("DF_API_URL", "http://localhost:3888/v1")
    api_key = os.getenv("DF_API_KEY")
    model = os.getenv("DF_MODEL", "gemini-3-flash-preview")

    state = SimpleNamespace(
        request=SimpleNamespace(
            chat_api_url=api_url,
            api_key=api_key,
            model=model,
        )
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹中的所有图像文件
    for img_name in tqdm(os.listdir(imgs_dir), desc="Processing images"):
        img_path = os.path.join(imgs_dir, img_name)

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # 跳过非图像文件

        print(f"Processing image: {img_name}")  # 打印每个被处理的文件

        # 初始化Mermaid代码
        mermaid_code = ""  

        mermaid_part = await analyze_image_data_flow(state, img_path)

        if mermaid_part:
            mermaid_code += f"{mermaid_part}"

        # 为每个图像生成一个单独的Mermaid代码文件
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.mmd")
        print(f"Outputting Mermaid code to: {output_file_path}")  # 打印输出的文件路径

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)

        print(f"Mermaid代码文件已输出: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析图像文件夹中的所有图像并生成Mermaid代码文件夹输出")
    parser.add_argument("--imgs", required=True, help="图像文件夹路径")
    parser.add_argument("--out", required=True, help="输出Mermaid代码文件夹路径")
    args = parser.parse_args()

    asyncio.run(process_image_and_generate_mermaid(args.imgs, args.out))

