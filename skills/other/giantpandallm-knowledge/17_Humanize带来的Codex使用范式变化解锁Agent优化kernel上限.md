# Humanize带来的Codex使用范式变化，解锁Agent优化kernel上限

**原文链接**: https://www.163.com/dy/article/KT0K0IRK05568W0A.html

**下载时间**: 2026-06-05 22:38:32

---

来源：GiantPandaLLM

## 0x0. 前言

之前做了几个模型profile调优相关的SKILL，还有Kernel优化相关的SKILL。本文讨论Humanize带来的Codex使用范式变化。

## 0x2. AVO和Codex /goal

AVO论文（Agentic Variation Operators for Autonomous Evolutionary Search）讨论的是把LLM从候选生成器升级成variation operator：Agent可以自己看历史版本、知识库和执行反馈，然后决定读什么、改什么、测什么、怎么修。

这篇论文里AVO在B200上连续跑7天，探索了500多个优化方向，提交了40个kernel版本，最后在MHA上比cuDNN最多快3.5%，比FlashAttention-4最多快10.5%。

## 0x4. KernelPilot

KernelPilot（https://github.com/BBuf/kernel-pilot）基于vendored upstream Humanize，再加kernel专用工作流：
- 新增humanize-kernel-agent-loop：用户只给一句kernel优化目标，它自己生成plan、refine plan、建standalone repo、启动RLCR
- 新增kernel-knowledge：本地GPU kernel知识库，先按topic/framework路由
- 新增profile-evidence：把NCU输出整理成Profile Evidence Digest
- 强制standalone repo：候选kernel不直接污染SGLang/vLLM大仓库
- 强制记录：attempt-ledger.md记录所有版本

## 0x6. int8_scaled_mm当前效果

在H100上，M=64,N=2048,K=2048,fp16,bias=true：
- v0 scalar：0.603872 ms
- SGLang baseline：0.017888 ms
- v23：0.015184 ms
- 从scalar到v23约39.8x加速
- 相比SGLang baseline快15.12%

## 0x7. 解锁更多模型优化的可能

SGLang这类推理框架的优化变成三件事：
- 人定义目标、边界和验收标准
- Agent负责长期执行、读代码、改代码、跑实验
- Harness负责记忆、审查、profile证据和停止条件