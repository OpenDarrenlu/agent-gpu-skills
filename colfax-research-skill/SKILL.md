---
name: colfax-research-skill
description: >
  Guide to Colfax Research articles on CUDA GPU optimization, CUTLASS, FlashAttention, and NVIDIA GPU architecture.
  Use when the user asks about: (1) CUDA kernel optimization, GEMM, or tensor operations on NVIDIA GPUs,
  (2) CUTLASS library usage, CuTe abstractions, or layout algebra,
  (3) FlashAttention implementations (FA-2, FA-3, FA-4), FlexAttention, or attention optimization,
  (4) NVIDIA GPU architectures (Hopper, Blackwell, Grace-Hopper),
  (5) FP8/INT8 quantization, mixed-precision training, or low-precision inference,
  (6) Cluster Launch Control (CLC), TMA, WGMMA, or other Hopper/Blackwell-specific features,
  (7) Any topic related to Colfax Research publications or GPU kernel performance tuning.
  The skill provides article recommendations with Chinese explanations from the complete Colfax Research knowledge base.
---

# Colfax Research Guide

Knowledge base of 29 Colfax Research articles (2023-2026) covering CUDA kernel optimization, CUTLASS, FlashAttention, and NVIDIA GPU architectures.

## Workflow

When user asks a question related to GPU/CUDA/CUTLASS/FlashAttention:

1. **Read article index**: Load `colfax_knowledge_base/metadata.json` to see all available articles with metadata.

2. **Recommend articles**: Based on the user's question, identify the most relevant articles by matching keywords against article titles, categories, and excerpts.

3. **Read article content**: For each recommended article, read its local markdown file at `colfax_knowledge_base/articles/{article_slug}.md` to get full content.

4. **Provide Chinese interpretation**: For each recommended article, provide:
   - **文章标题** (Article Title)
   - **核心内容** (Key Content): 1-2 sentence summary in Chinese
   - **为什么相关** (Why Relevant): Explain how this article addresses the user's question
   - **原文链接** (Original Link): The article URL
   - **PDF availability**: Note if a PDF version is available for download

5. **Suggest reading order**: If multiple articles are relevant, suggest the optimal reading sequence based on prerequisite knowledge.

## Article Categories

| Category | Count | Description |
|----------|-------|-------------|
| CUTLASS Tutorials | 11 | Step-by-step guides for writing optimized GEMM kernels |
| FlashAttention | 7 | FlashAttention-2/3/4 implementations and optimizations |
| CuTe/Layout Algebra | 4 | Mathematical foundations of CUTLASS layout abstractions |
| GPU Architecture | 4 | Hopper/Blackwell-specific features and system guides |
| Deep Learning | 3 | Mixed-precision training and inference optimization |

## Key Topics Covered

- **GEMM Optimization**: Tiling, pipelining, WGMMA, TMA, persistent kernels, Stream-K
- **Attention Kernels**: FlashAttention-2/3/4, FlexAttention, online-softmax fusion
- **Low Precision**: FP8, INT8, sub-byte types, block-scaling, quantization
- **GPU Architectures**: Hopper H100, Blackwell B200, Grace-Hopper Superchip
- **CUTLASS APIs**: CuTe DSL, layout algebra, epilogue fusion, epilogue visitor trees

## Incremental Update

To update the knowledge base with new articles from Colfax Research:

```bash
python3 scripts/update_kb.py
```

This script checks for new articles and downloads them incrementally without re-fetching existing content.
