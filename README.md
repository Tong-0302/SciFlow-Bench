# SciFlow-Bench

Official repository for the paper: **"SciFlow-Bench: Evaluating Structure-Aware Scientific Diagram Generation via Inverse Parsing"**

*Tong Zhang\*, Honglin Lin\*, Zhou Liu, Chong Chen, Wentao Zhang†*

---

**SciFlow-Bench** is a benchmark for evaluating structure-aware scientific diagram generation via inverse parsing. It covers **500 papers** across 5 research domains and evaluates generated images at three complementary levels: graph structure, text accuracy, and visual quality.

## Overview

Given a framework figure from an academic paper, SciFlow-Bench:
1. **Inverse Parsing** — extracts structured graph representations (nodes + edges) via a 21-step automated pipeline
2. **Generation** — reproduces diagrams using various baseline models
3. **Structure-Aware Evaluation** — evaluates the generated images against ground truth at three levels

### Evaluation Metrics

| Level | What it measures | Key metrics |
|-------|-----------------|-------------|
| **Graph-level** | Structural correctness (nodes & edges) | Semantic Node F1, Edge F1 |
| **Text-level** | Text content accuracy in the figure | Embedding similarity, LLM-based scoring |
| **Image-level** | Visual fidelity | CLIP Score, LPIPS, VLM-based assessment |

Final score = 0.4 × Graph + 0.3 × Text + 0.3 × Image

### Benchmark Coverage

| Domain | Papers |
|--------|--------|
| Computer Vision (CV) | 175 |
| Natural Language Processing (NLP) | 122 |
| Machine Learning (ML) | 75 |
| Integrated Circuits (IC) | 75 |
| Robotics | 53 |
| **Total** | **500** |

## Project Structure

```
AutoBench/
├── autobench_index.json          # Paper index (arXiv IDs per domain)
├── pipeline/                     # Benchmark construction pipeline
│   ├── run_pipeline.py           # Main entry point
│   ├── run_batch_pipeline.py     # Batch processing
│   └── step01-step21_*.py        # 21-step extraction pipeline
├── evaluation/                   # Evaluation system
│   ├── evaluate.py               # Main evaluation entry
│   ├── eval_graph_level.py       # Graph structure metrics
│   ├── eval_text_level.py        # Text accuracy metrics
│   └── eval_image_level.py       # Image quality metrics
├── baselines/                    # Baseline model implementations
│   ├── baseline_generate.py      # Generation entry
│   ├── baseline_pipeline.py      # Pipeline runner
│   └── models/                   # Model wrappers
│       ├── graphviz_gen.py       # Graphviz (code-based)
│       ├── sdxl.py               # Stable Diffusion XL
│       ├── flux.py / flux_dev.py # FLUX.1 / FLUX.2-dev
│       ├── pixart.py             # PixArt
│       ├── qwen.py               # Qwen-Image
│       ├── seedream.py           # Seedream
│       └── gemini_dalle.py       # Gemini + DALL-E
└── utils/                        # Shared utilities
    ├── llm_callers/              # LLM API wrappers
    ├── logger.py
    └── state.py
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/AutoBench.git
cd AutoBench
pip install -r requirements.txt
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `DF_API_KEY` — API key for LLM services
- `DF_API_URL` — API endpoint URL

## Usage

### 1. Run the Benchmark Construction Pipeline

Process a paper PDF into a structured graph:

```bash
python -m pipeline.run_pipeline \
    --pdf /path/to/paper.pdf \
    --output /path/to/output_dir
```

Batch process multiple papers:

```bash
python -m pipeline.run_batch_pipeline \
    --index autobench_index.json \
    --output /path/to/output_dir
```

### 2. Generate Baseline Images

```bash
python -m baselines.baseline_generate \
    --model seedream \
    --input /path/to/graph.json \
    --output /path/to/generated_images
```

Supported models: `graphviz`, `sdxl`, `flux`, `pixart`, `qwen`, `seedream`, `gemini_dalle`

### 3. Run Evaluation

```bash
python -m evaluation.evaluate \
    --gt /path/to/ground_truth \
    --pred /path/to/predictions \
    --output results.json
```

## Pipeline Steps

The 21-step pipeline transforms a paper PDF into a structured graph:

| Step | Description |
|------|-------------|
| 01 | PDF OCR inference |
| 02 | Select framework figure |
| 03 | Extract method & figure titles |
| 04 | Detect layout |
| 05 | SAM segmentation |
| 06 | Compute canvas size |
| 07-09 | Chunk segmentation & summarization |
| 10 | Select best segments |
| 11 | Describe and filter nodes |
| 12-14 | Remove artifacts & filter invalid nodes |
| 15-17 | Parse output & remove OCR artifacts |
| 18 | Merge results |
| 19 | Generate Mermaid representation |
| 20 | Extract edges |
| 21 | Assemble final graph JSON |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2025sciflowbench,
  title={SciFlow-Bench: Evaluating Structure-Aware Scientific Diagram Generation via Inverse Parsing},
  author={Zhang, Tong and Lin, Honglin and Liu, Zhou and Chen, Chong and Zhang, Wentao},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
