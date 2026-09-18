# Mamba Kernel详解

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/691365497

---

工作之余看了mamba 的论文，对论文的内容有一些一知半解，所以又阅览了一下mamba的源码，本文大致总结了mamba的结构和CUDA kernel实现。

mamba的主要结构是融合了H4 block + Gateed MLP，通过引入Selecitve State-Space Model （SSM）来代替transformer中的Attention Block，通过SSM来解决attention 中 
𝑂
(
𝑁
2
)
 的复杂度，比避免了存储完整的上下文（KV Cache）。

SSM的结构可以用如下的迭代公式表达
ℎ
𝑡
=
𝐴
¯
ℎ
𝑡
−
1
+
𝐵
¯
𝑢
𝑡
𝑦
𝑡
=
𝐶
ℎ
𝑡

在后文中，用D=d_model, N=d_inner, H=d_state来指代，如上图A第n个元素分别用 \frac{1}{2}+ni 和 -（n+1） 来初始化复数和实数参数， Bias_{\Delta} = \tau^{-1}(Uniform([0.001,0.1]) ， D=\mathbf{1}^{N} 。阅读源码中 \bar{B} = \Delta B 和论文中的计算方式不一致，有点疑问。

Mamba Block 结构
mamba block
mamba block的结构整体与Gated MLP类似，input 进行两次Linear 层得到 x和z。对x的L（seq）维度进行conv1d和silu得到u，并对u进行Linear project 得到delta、B、C，再对delta 进行low-rank project。 将A、B、C、D、delta、u、 Bias_{\Delta} 、z作为Selective_SSM的输入
Selective-SSM 伪代码：

Selective SSM的核心是通过Inclusive_scan, 在GPU 上进行多线程并行递归。 thread\_data_{i} = [\bar{A}_{i}, \bar{B}_{i}u_{i}]\\ Scan\_Op([a, b], [c,d]) = [ac, c*b+d]\\ thread\_data_{i}' =Inclusive\_Scan(thread\_data_{0...i}, Scan\_Op)\\

Selective-SSM Cuda Kernel 实现

若seq<=1024, 每个cta launch kNThreads， 每个thread 负责计算seq中 kNItems\times kNROW 元素，若seq>1024, 则以 2048=（128(kNThreads)\times16(kNItems)） 为chunk_size进行循环。若 seq\%(kNThreads\times kNItems)=0 ，避免了数据尾块的情况，则通过vectorization 进行数据读取提高访存带宽。

Selective SSM Kernel 完全基于CUB Module 中的IO Block module 和Scan Block Module 进行实现

thread grid (batch, \frac{N}{KNROW}) , thread block (KTHREAD)

每个thread block 负责计算 output\in \mathbb{R}^{B\times N\times L} 中的 output_{CTA}\in \mathbb{R}^{KNROW \times L} ， 如上图红色部分。

每个thread 负责计算 output_{CTA}\in \mathbb{R}^{KNROW \times L} 中的 output_{thread}\in \mathbb{R}^{KNROW \times KNItems} ，如上图橙黄色部分

selective-ssm 核心计算循环
