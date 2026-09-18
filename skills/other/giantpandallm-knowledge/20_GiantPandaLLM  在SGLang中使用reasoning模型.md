# GiantPandaLLM | 在SGLang中使用reasoning模型

**原文链接**: https://blog.csdn.net/csdn_xmj/article/details/147929348

**下载时间**: 2026-06-05 22:38:32

---

GiantPandaLLM | 在SGLang中使用reasoning模型(建议收藏！)

## 基本推理

使用SGLang启动reasoning模型（如Qwen3-8B），可以获取reasoning_content和content分离的响应。

## 解析推理内容

可以分别获取模型的思考过程（reasoning）和最终答案（text）。

## 结构化输出

使用Pydantic以方便的方式制定结构，让模型输出JSON等结构化数据。

示例：
```
reasoning content: Okay, the user is asking for the name and population of the capital of France...
content: {"name": "Paris", "population": 2100000}
```

更多详细信息请参阅SGLang文档：https://docs.sglang.ai/backend/separate_reasoning.html