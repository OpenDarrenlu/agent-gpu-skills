# GiantPandaLLM | 非常简洁的图像复原新方法：退化分类预训练

**原文链接**: https://blog.csdn.net/csdn_xmj/article/details/146249416

**下载时间**: 2026-06-05 22:38:32

---

GiantPandaLLM | 非常简洁的图像复原新方法：退化分类预训练，已中ICLR2025

论文地址：https://openreview.net/forum?id=PacBhLzeGO
代码地址：https://github.com/mc-lan/Clear2Rainy

## 核心发现

1. 随机初始化模型显示出对退化进行分类的内在能力
2. 在一体化（All-in-one）复原任务中训练的模型表现出辨别未知退化的能力
3. 在修复模型的早期训练中，有一个退化理解步骤

## 预实验结果

提取了复原训练过程中网络复原头之前的输出特征，训练过程中模型仅见到雾霾、雨、高斯噪声三种退化。根据该特征，kNN分类器将对五种退化类型（包括雾霾、雨天、高斯噪声、运动模糊和弱光）进行分类。

|Methods|NAFNet|SwinIR|Restormer|PromptIR|
|-|-|-|-|-|
|Acc. on Random initialized (%)|52 ± 1|64 ± 4|71 ± 4|55 ± 3|
|Acc. on 3D all-in-one trained 200k iterations (%)|90 ± 5|92 ± 6|93 ± 3|93 ± 5|
|Acc. on 3D all-in-one trained 400k iterations (%)|94 ± 4|95 ± 4|95 ± 4|95 ± 4|
|Acc. on 3D all-in-one trained 600k iterations (%)|94 ± 5|95 ± 4|97 ± 2|95 ± 4|

## 核心结论

复原中隐藏着（退化）辨别。在复原模型中加入卓越的降解感知判别信息，并最大限度地提高其判别能力，将进一步提高模型的复原性能。