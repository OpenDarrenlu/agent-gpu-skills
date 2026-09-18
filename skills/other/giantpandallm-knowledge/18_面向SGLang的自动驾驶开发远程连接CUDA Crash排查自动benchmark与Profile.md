# 面向SGLang的自动驾驶开发：远程连接、CUDA Crash排查，自动benchmark与Profile分析

**原文链接**: https://finance.sina.com.cn/wm/2026-04-01/doc-inhtaktw2132411.shtml

**下载时间**: 2026-06-05 22:38:32

---

来源：GiantPandaLLM

## 0x0. 前言

这篇文章整理了笔者近一段时间高频使用的几个SGLang相关SKILL，内容覆盖debug、benchmark、远程开发以及性能分析等场景。

## 0x1. 远程连接SKILL

- H100远端skill
- B200远端skill
- H200 diffusion远端skill

作用：很多SGLang验证工作依赖远程GPU服务器，包括模型加载、kernel smoke test、端到端服务验证、benchmark、profiler采集等。

## 0x2. SGLang CUDA Debug Crash SKILL

放在：https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash

## 0x3. SGLang Auto-Driven Benchmark SKILL

放在：https://github.com/sgl-project/sglang/pull/21736

## 目标

让更多SGLang相关工作逐步进入Agent可自动完成、自动分析和自动优化的流程。