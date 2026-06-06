---
name: zhihu-knowledge-base
description: 检索和总结用户关注的知乎博主文章知识库。当用户询问技术问题、需要查找特定知识点、或提到"知乎文章"、"我关注的博主"、"知识库"时，使用此Skill搜索本地索引的知乎文章并提供总结。
---

# 知乎知识库检索 Skill

本Skill用于检索用户关注的258位知乎博主的文章知识库（共1028篇文章，约612万字），并根据用户问题找到最相关的文章进行总结。

## 知识库位置

- 文章存储: `/Users/moonshot/Documents/kimi/workspace/zhihu_knowledge_base/articles/`
- 索引数据库: `/Users/moonshot/Documents/kimi/workspace/zhihu_knowledge_base/index/zhihu_articles.db`
- 搜索脚本: `/Users/moonshot/Documents/kimi/workspace/zhihu_knowledge_base/scripts/zhihu_search.py`

## 使用流程

### 1. 搜索相关文章

当用户提出技术问题时，先用搜索脚本找到相关文章：

```bash
python3 /Users/moonshot/Documents/kimi/workspace/zhihu_knowledge_base/scripts/zhihu_search.py search "<关键词>" <数量>
```

关键词建议：
- 提取用户问题中的核心技术术语
- 可尝试多个关键词组合搜索
- 常用技术词：CUDA, GPU, TPU, LLM, Transformer, FlashAttention, CUTLASS, Triton, NCCL, MoE, DeepSeek, PyTorch, TensorRT, GEMM, 编译器, 体系结构, 量化, 推理, 训练

### 2. 读取文章全文

获取最相关文章的完整内容：

```bash
python3 /Users/moonshot/Documents/kimi/workspace/zhihu_knowledge_base/scripts/zhihu_search.py content <article_id>
```

### 3. 总结核心思想

基于搜索到的文章，为用户总结：
- 核心观点和结论
- 关键技术细节
- 与问题的直接关联
- 引用文章来源（作者+标题+链接）

## 搜索策略

1. **直接搜索**: 用用户问题的核心关键词搜索
2. **多关键词尝试**: 如果首次搜索结果不理想，尝试同义词或相关技术词
3. **作者筛选**: 如果用户提到特定博主，先用 `by-author` 命令列出该博主文章
4. **组合阅读**: 有时需要综合多篇文章的观点给出完整答案

## 注意事项

- 知识库覆盖领域：AI/ML、GPU/CUDA、计算机体系结构、编译器、深度学习框架、推理优化等
- 文章均为用户关注的知乎博主原创，引用时请标注作者和原文链接
- 如果知识库中没有直接相关文章，诚实告知用户并尝试用通用知识回答
