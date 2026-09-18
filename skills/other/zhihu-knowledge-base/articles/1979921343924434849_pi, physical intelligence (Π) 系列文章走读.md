# pi, physical intelligence (Π) 系列文章走读

**作者**: 小志哥​​上海引望智能技术有限公司 员工

**原文链接**: https://zhuanlan.zhihu.com/p/1979921343924434849

---

Overview

主页：https://www.pi.website/

未完待续

名词解释
ITEM	Explanation
out-of-the-box evaluation	使用非定制化的模型，在定制化的任务上进行测试验证
	
	
	
pi 0.6 Model Card

https://website.pi-asset.com/pi06star/PI06_model_card.pdf

输入
four images, 448x448。内部使用双向注意力
text prompt
tokenized 自体运动传感器信息。
(optional) conditioning metadata?
噪声（用于action expert）
模型
VLM，Gemma3, 4B
860M, flow-matching action expert
输出
文字输出
离散动作输出
底层级 - 连续动作输出（由action expert输出）。内部使用双向注意力，梯度不回传到VLM。
推理性能
实时系统，With 5 denoising steps and 3 camera inputs, π0.6 takes 63ms to produce an action chunk on a single H100 GPU




pi 0.6 *

blog: https://www.pi.website/blog/pistar06

论文：https://arxiv.org/pdf/2511.14759，一堆公式

代码未公开

学习路径：

模仿好行为
缺点：累积误差；错误相对稀疏；因果混淆；
从坏行为里学习
人类专家指出并修复错误。
缺点：成本高；小错误不敏感；无法做到最优
自我探索




Recap(RL with Experience & Corrections via Advantage-conditioned Policies)




怎么应对长时间序列？没说
将advantage作为条件时，RL怎么通过分数奖励来优化policy？这部分看起来更类似模仿学习，没有一个仿真器来提供奖励。
训练过程？
pretrain：offline RL。用各种演示数据（应该包括了差数据？），训value model，再训advantage conditioned policy model。真离线。
value model？ 论文里叽里呱啦说了一堆公式，说value 是t到结束的reward总和，但从实现来看，value被定义为简单的距离结束的步数。
fine-tune: with specific tasks
RL: 用接管数据、奖励模型训？ 认为接管数据都是好动作，value=positive







背景知识
Classifier Guidance / Classifier Free Guidance

diffusion生成时，引导模型生成指定语义的内容，用的两种方法。后者就是我们常见的有条件的生成。

Classifier Guidance 和 Classifier Free Guidance，一堆公式不如两行代码
