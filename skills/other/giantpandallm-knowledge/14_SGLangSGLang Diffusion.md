# SGLang｜SGLang Diffusion

**原文链接**: https://mp.weixin.qq.com/s/lnCp1FXz71s_yhe0IS0jzQ

**下载时间**: 2026-06-05 22:38:32

---

GiantPandaLLM

来自 SGLang 开源社区的 Yichi 介绍了 **SGLang Diffusion**，一个面向图像与视频生成的高性能、全开源扩散模型推理引擎。

## 目标

在保持灵活性和生产可用性的同时，大幅提升文本生成图像、文本生成视频、图像生成图像、图像生成视频等工作流的速度。

## 特点

- 基于 SGLang 已经过验证的调度机制与内核级优化
- 支持多种扩散模型与生成模式
- 开源的 API、命令行工具和 Python 绑定
- 与 Hugging Face Diffusers 的性能对比（最佳情况下最高可快 5.9 倍）
- 基于 CFG 并行的多 GPU 加速能力
- 支持文本生成视频、图像到图像风格迁移、图像到视频动画等

## 技术博客

https://lmsys.org/blog/2025-11-07-sglang-diffusion/