# 记录下SGLang开发，优化，debug的技巧之大SKILL时代已来临

**原文链接**: https://k.sina.com.cn/article_5952915720_162d2490806703jsuk.html

**下载时间**: 2026-06-05 22:38:32

---

来源：GiantPandaLLM

## 0x0. 前言

之前在《记录下SGLang开发，编译和Profile的几个小技巧》和《记录下SGLang开发，debug的几个技巧第二弹》中，记录了一些SGLang开发、debug、profile的技巧。

## 0x1. Agent的冲击

在经历了Codex + GPT5.4 Extra High狂蹬2周做的事情之后，觉得之前自己的学习基本失去意义，一些难理解的知识和总结的技巧，其实只是大模型在设置合适context（SKILL）下的Token而已。

Codex + GPT5.4已经达到了非常强的能力，这和2025年的感觉完全不一样，真正的智能似乎已经出现了，至少在编程开发领域是这样。

## 实际成果

读者在Codex或者Claude Code中可以安装SGLang提供的一些SKILLS，完成：
- kernel编写
- benchmark和测试编写
- kernel迭代优化
- 模型编写
- 模型优化
- CUDA Crash自动debug
- 自动二分坏掉的commit

最近基于Codex和这些SKILL，让SGLang Diffusion的Z-Image单卡速度提升40%，Qwen/Qwen-Image-2512的单卡速度提升20%+，并挖掘了一个kernel fuse的pattern。

如果用AKO4ALL（https://github.com/TongmingLAIC/AKO4ALL），可以让已有kernel更容易获得提升。