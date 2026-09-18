# 维度6：ML辅助的GPU性能建模方法

## 维度概述

近年来，机器学习方法被广泛应用于GPU性能预测领域，形成了从简单的线性回归到复杂的神经网络等多种方法论。ML辅助的GPU性能建模旨在通过学习历史性能数据来预测未见过的GPU架构或工作负载的执行延迟，从而指导硬件选型、编译器优化和分布式部署决策。本维度深入调研了当前主流的ML辅助GPU性能建模方法，包括NeuSight的tile粒度预测、PipeWeave/SYNPERF的分析-ML协同框架、Habitat的混合预测、AutoTVM/Ansor的梯度提升树cost model、WaveTune的wave-aware bilinear建模等，系统分析各方法的模型架构、输入特征、训练策略、泛化能力及误差表现。

---

## 核心发现

### 1. NeuSight：Tile-粒度的Utilization预测框架

**Claim:** NeuSight通过将端到端延迟预测分解为tile粒度的子问题，利用轻量MLP预测GPU utilization（而非直接预测延迟），并结合fundamental performance laws进行bounding，在H100上实现GPT3推理延迟预测误差从121.4%降至2.3%。[^3^][^4^][^62^]

**Source:** Lee et al., "Forecasting GPU Performance for Deep Learning Training and Inference," arXiv:2407.13853, 2024.

**URL:** https://arxiv.org/abs/2407.13853

**Date:** 2024-07-18

**Excerpt:** "We develop NeuSight which digresses from prior works as it leverages the insight that popular GPU libraries decompose deep learning kernels into multiple smaller working sets. These small working sets, or tiles, are then dispatched and executed on the GPU independently across the SMs. Based on this observation we partition the end-to-end latency prediction into regular and more manageable sub-problems, predicting the latency at tile-granularity."

**Context:** NeuSight的核心创新在于不直接用MLP预测延迟，而是预测GPU utilization（设备利用率），然后通过performance bounds（峰值FLOPS和内存带宽）来约束预测结果，确保预测的合理性。

**Confidence:** high

---

**Claim:** NeuSight的MLP架构包含8个隐藏层、每层512个单元，使用ReLU激活函数，为5种不同算子（Batched MatMul、FC、Element-wise、Softmax、LayerNorm）分别训练独立的MLP。输入特征经过per-SM归一化和utilization预处理后 feeding给MLP。[^3^][^7^]

**Source:** Lee et al., "Forecasting GPU Performance for Deep Learning Training and Inference," arXiv:2407.13853, 2024.

**URL:** https://arxiv.org/html/2407.13853v2

**Date:** 2024-01-18

**Excerpt:** "Each MLP comprises 8 hidden layers, each with 512 hidden units, similar to prior work (Yu et al., 2021). First-layer converts the input feature vectors into 512-dimensional hidden vector, and the last layer converts the 512-dimensional hidden vector to single-dimensional output. ReLU is used as the activation function and applied at the end of every layer."

**Context:** NeuSight使用AdamW优化器，L2正则化，训练100 epochs，batch size从16到128不等，学习率范围1e-6到5e-3。损失函数使用Symmetric Mean Absolute Percentage Error (sMAPE)。

**Confidence:** high

---

**Claim:** NeuSight在单设备推理和分布式训练中均表现出色。在H100上预测GPT3推理误差为2.3%，训练误差8.9%；在4-GPU服务器上分布式训练预测误差5.4%。对operator fusion的支持误差约15.7%，对FP16和Tensor Core的泛化误差约13%。[^2^][^3^]

**Source:** Lee et al., arXiv:2407.13853v3, 2024.

**URL:** https://arxiv.org/html/2407.13853v3

**Date:** 2024-12-12

**Excerpt:** "The evaluation on a diverse set of GPUs (Nvidia H100, A100-80GB, V100, A100-40GB, P100, T4, L4) and deep learning workloads (BERT, GPT2, GPT3, OPT, Switch Transformer) shows that NEUSIGHT demonstrates a percentage error of 8.9%, compared to 140% by MLP-based and 60.8% by linear regression-based prior work."

**Context:** 对比实验显示，NeuSight显著优于Habitat（MLP直接预测延迟）和线性回归方法，尤其在unseen GPU（如H100、L4）和unseen model dimension上表现突出。

**Confidence:** high

---

### 2. PipeWeave/SYNPERF：分析模型与ML的协同设计

**Claim:** PipeWeave（即SYNPERF）提出了知识驱动（分析模型分解）与数据驱动（轻量MLP）协同的GPU性能预测框架，通过将kernel分解为fundamental tasks、模拟SM调度、分析pipeline demand，最后以轻量MLP预测执行效率。在seen GPU上MAPE 6.1%，unseen GPU上11.4%，相对NeuSight误差降低6.7x。[^1^][^12^]

**Source:** "PipeWeave: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction," arXiv:2601.14910, 2025.

**URL:** https://arxiv.org/abs/2601.14910

**Date:** 2025-10-27

**Excerpt:** "We present PipeWeave, a framework that achieves high fidelity, fast speed, and broad generalizability in GPU performance modeling through a combined analytical-ML design that weaves pipeline-level analysis into accurate predictions."

**Context:** PipeWeave的四个核心模块：Kernel Decomposer、Scheduling Simulator、Feature Analyzer、Performance Estimator。MLP仅用于捕获高阶非线性交互，主要工作由分析模型完成。

**Confidence:** high

---

**Claim:** PipeWeave的MLP采用3层隐藏层（256/128/64单元），ReLU激活+BatchNorm+Dropout(0.1)，输出层使用Sigmoid预测execution efficiency（理论时间与实际延迟的比值）。使用AdamW优化器，学习率0.001，MAPE作为损失函数。[^1^]

**Source:** PipeWeave arXiv:2601.14910v2, Section V-C.

**URL:** https://arxiv.org/html/2601.14910v2

**Date:** 2025-10-27

**Excerpt:** "The MLP has a shallow architecture with 3 hidden layers (256, 128, and 64 units), employing ReLU activations followed by Batch Normalization and Dropout (rate 0.1) for regularization. The output layer utilizes a Sigmoid activation to limit predictions to the range [0,1], representing the kernel's execution efficiency."

**Context:** 这种浅层MLP设计相比NeuSight的8层512单元网络更轻量，且因为分析模型已经捕获了一阶效应，MLP只需要学习剩余的高阶交互。

**Confidence:** high

---

**Claim:** 在E2E推理场景中，PipeWeave在seen GPU上平均误差8.5%，unseen GPU上10.7%。在SGLang和vLLM框架上测试Qwen2.5-14B、Qwen3-32B、Llama3.1-70B模型，所有20种分布式配置下平均MAPE 6.6%，而NeuSight为34.7%（5.3x差距）。[^1^][^12^]

**Source:** PipeWeave arXiv:2601.14910.

**URL:** https://arxiv.org/pdf/2601.14910

**Date:** 2025

**Excerpt:** "Across all 20 tested configurations, SYNPERF achieves an overall average MAPE of 6.6% versus Neusight's 34.7%, showing a 5.3x average accuracy improvement."

**Context:** PipeWeave还展示了"beyond simulation"的价值：通过quantile regression定义性能ceiling，识别出Fused MoE Triton kernel中的实现低效性，指导优化实现1.7x加速。

**Confidence:** high

---

### 3. Habitat：Kernel分类与混合预测

**Claim:** Habitat将算子分为kernel-alike（跨GPU使用相似kernel）和kernel-varying（跨GPU使用不同kernel）两类。对kernel-alike算子使用wave scaling技术基于内存带宽和计算单元比例进行缩放；对kernel-varying算子（conv2d、LSTM、BMM、Linear）使用预训练MLP预测延迟。平均预测误差11.8%。[^52^][^168^]

**Source:** Yu et al., "Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training," ATC 2021.

**URL:** https://www.cs.toronto.edu/ecosystem/papers/ATC_21/Habitat.pdf

**Date:** 2021

**Excerpt:** "Habitat classifies operators into two types: kernel-alike which use similar kernels across different GPU architectures, and kernel-varying which use different kernels across different GPU architectures."

**Context:** Habitat需要在一台参考GPU上执行一次训练迭代，记录运行时信息后预测其他GPU上的性能。这是一种hybrid runtime-based approach。

**Confidence:** high

---

**Claim:** Habitat的MLP为每种kernel-varying算子独立训练，输入特征包括：层维度（如conv的输入/输出通道）、目标GPU的内存容量和带宽、SM数量、峰值FLOPS。MLP架构为8个隐藏层，每层1024单元，ReLU激活。[^52^]

**Source:** Yu et al., "A Runtime-Based Computational Performance Predictor for Deep Neural Network Training," arXiv:2102.00527.

**URL:** https://arxiv.org/pdf/2102.00527

**Date:** 2021

**Excerpt:** "Each MLP comprises an input layer, eight hidden layers, and an output layer that produces a single real number—the predicted execution time. We use ReLU activation functions in each layer and we use 1024 units in each hidden layer."

**Context:** Habitat中95%的unique operations用wave scaling预测（占执行时间的46%），5%用MLP预测（占执行时间的54%），展示了混合方法的有效性。

**Confidence:** high

---

**Claim:** Habitat在泛化到更大矩阵维度（>1024）和新GPU时面临挑战，误差可达38%；在A100等unseen GPU上误差可达127%。这主要是因为其MLP直接建模整体延迟，且训练数据仅覆盖2018年前的GPU。[^7^][^53^]

**Source:** Lee et al., "Data-driven Forecasting of Deep Learning Performance on GPUs," arXiv:2407.13853v1, 2024.

**URL:** https://arxiv.org/html/2407.13853v1

**Date:** 2024-07-18

**Excerpt:** "Habitat faces challenges in generalizing to matrix multiplications with dimensions larger than 1024, resulting in a percentage error of up to 38%. Additionally, Habitat struggles with unseen GPUs, displaying a percentage error of up to 127% on A100."

**Context:** Habitat的局限性在于：(1) MLP直接预测延迟而非utilization；(2) 需要参考GPU执行；(3) 训练数据未覆盖新架构。

**Confidence:** high

---

### 4. AutoTVM/Ansor与TenSet：基于梯度提升树的Cost Model

**Claim:** AutoTVM和Ansor使用XGBoost（梯度提升树）作为cost model来预测tensor program的相对性能排名（而非绝对延迟）。XGBoost在AutoTVM中通过预测relative throughput来指导搜索，相比随机搜索显著减少测量次数。Ansor通过进化算法生成candidates，再用cost model筛选高质量programs。[^229^][^232^]

**Source:** Zheng et al., "Ansor: Generating High-Performance Tensor Programs for Deep Learning," OSDI 2020; Chen et al., "Learning to Optimize Tensor Programs," NeurIPS 2018.

**URL:** https://arxiv.org/pdf/2006.06762

**Date:** 2020/2018

**Excerpt:** "AutoTVM (Chen et al., 2018) and Ansor (Zheng and others, 2020) employed XGBoost to predict relative performance rankings."

**Context:** AutoTVM的cost model是一种GBDT（Gradient Boosted Decision Tree），输入包括从tensor program提取的164+特征（计算特征、内存访问特征、算术强度等），输出是normalized throughput。XGBoost用于ranking任务，在搜索过程中快速评估候选program的优劣。

**Confidence:** high

---

**Claim:** TenSet是一个大规模tensor program性能数据集，包含来自120个神经网络、6个硬件平台的超过5200万条测量记录。TenSet MLP在Ansor中预训练后可将搜索时间加速最多10倍。MLP架构为4层，使用RankLoss训练以预测relative performance。[^232^]

**Source:** Zheng et al., "TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers," NeurIPS 2021 Datasets and Benchmarks Track.

**URL:** https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf

**Date:** 2021

**Excerpt:** "The dataset consists of over 13,000 tasks from 120 networks measured on six hardware platforms... resulting in over 52 million measurements."

**Context:** TenSet的特征提取包含多个层次：高级task描述特征（shape、access pattern）、优化特征（loop transformations、schedules）、低级程序特征（lowered IR、machine code）。硬件特征包括cache size、vector width、memory bandwidth等。

**Confidence:** high

---

### 5. TLP：基于Transformer的Cost Model

**Claim:** TLP（Tensor Program Tuning via Learning-based Prediction）是一种基于Transformer的cost model，使用self-attention机制从schedule primitive序列中提取特征，相比TenSet MLP在CPU上top-1 score大幅提升，在GPU上也有竞争力。TLP的搜索速度比TenSet MLP快3-9倍。[^227^][^230^]

**Source:** Zhai et al., "TLP: A Deep Learning-based Cost Model for Tensor Program Tuning," ASPLOS 2023.

**URL:** https://arxiv.org/pdf/2211.03578

**Date:** 2023

**Excerpt:** "We use the self-attention or LSTM module to capture contextual features... self-attention+lambda rank loss is slightly more suitable for TLP than the other combinations."

**Context:** TLP的模型架构：浅层线性层将embedding size从22 upsample到256，8-head single-layer self-attention，后跟2个residual blocks，最后线性层+sum得到prediction score。使用lambda rank loss训练。相比TenSet MLP从tensor program提取特征，TLP从schedule primitives提取特征，速度更快且更通用。

**Confidence:** high

---

**Claim:** TLP的自注意力机制能够有效捕获schedule primitives之间的长距离依赖关系，这是传统MLP和XGBoost难以做到的。TLP使用相同的特征提取机制处理CPU和GPU，而TenSet MLP需要为不同硬件平台做特殊特征工程。[^227^][^230^]

**Source:** Zhai et al., TLP, ASPLOS 2023.

**URL:** https://arxiv.org/pdf/2211.03578v2

**Date:** 2023

**Excerpt:** "TenSet MLP performs special feature extraction for CPUs and GPUs, and TLP uses the same mechanism to extract CPU and GPU features. We can say that TLP feature extraction is a simple yet effective and general method."

**Context:** TLP的核心思想转变：不从tensor program本身提取特征，而是从schedule primitives提取特征。这使得TLP不受限于先验知识，也不需要提前prune特征。

**Confidence:** high

---

### 6. WaveTune：Wave-aware Bilinear Runtime Kernel配置

**Claim:** WaveTune发现GPU kernel延迟遵循wave-conditioned piecewise bilinear模式：在相同wave count下，延迟对grid size和loop count呈双线性关系（R^2 > 0.998）。基于这一发现，WaveTune构建了wave-aware分段双线性模型，实现运行时kernel配置的微秒级决策开销（5-6 us），相比穷举搜索降低5个数量级。[^49^][^91^][^153^]

**Source:** Zhang et al., "WaveTune: Wave-aware Bilinear Modeling for Efficient GPU Kernel Auto-tuning," arXiv:2604.10187, 2026.

**URL:** https://arxiv.org/abs/2604.10187

**Date:** 2026-04-11

**Excerpt:** "We observe that kernel latency follows a wave-conditioned piecewise bilinear pattern, where discrete execution waves introduce structured discontinuities, while intra-wave behavior remains approximately linear."

**Context:** WaveTune的核心模型公式：T(G,L) ≈ alpha*G*L + beta*G + gamma*L + delta，其中G为grid size，L为loop count。alpha*G*L项捕获核心的aggregate workload volume驱动的时间，beta*G和gamma*L分别建模block scheduling和per-iteration的开销。

**Confidence:** high

---

**Claim:** WaveTune在三个representative kernels（FlashAttention、Grouped GEMM、Dense GEMM）和五个GPU架构上评估，相比default heuristic实现最高1.83x kernel-level speedup和1.33x end-to-end TTFT reduction。运行时决策开销仅5-6 us，而XGBoost-based cost model需要1822-2965 us。[^91^][^153^]

**Source:** Zhang et al., WaveTune, arXiv:2604.10187, 2026.

**URL:** https://arxiv.org/html/2604.10187v1

**Date:** 2026-04-11

**Excerpt:** "WaveTune achieves near-oracle execution performance with a tightly bounded runtime overhead of 5-6 us across all workloads, enabled by its O(1) dual-table lookup and lightweight bilinear evaluation."

**Context:** WaveTune的dual-table设计：(1) Coefficient Table：按<macro-config, wave count>索引，存储bilinear参数元组；(2) Micro-config Table：按<macro-config, wave count, L>索引，缓存最优micro-config。运行时分为两个阶段：Stage I通过table-driven latency prediction选择macro-config；Stage II通过proximal loop anchor lookup检索micro-config。

**Confidence:** high

---

### 7. PM2Lat：基于SIMT架构特性的轻量预测

**Claim:** PM2Lat不依赖深度神经网络，而是利用GPU的SIMT（Single-Instruction-Multiple-Thread）架构特性，通过kernel differentiation和FLOPs-延迟的rational correlation建模。在Transformer模型上误差低于10%，在BF16数据类型上比NeuSight优至少50%。单次预测仅0.045ms（CPU），而NeuSight需6.5ms（GPU）。[^50^][^92^]

**Source:** La et al., "PM2Lat: Highly Accurate and Generalized Prediction of DNN Execution Latency on GPUs," arXiv:2603.00549, 2026.

**URL:** https://arxiv.org/abs/2603.00549

**Date:** 2026-02-28

**Excerpt:** "Unlike prior methods that rely on deep learning models or handcrafted heuristics, PM2Lat leverages the Single-Instruction-Multiple-Thread architecture of GPUs to model execution time of DNN models."

**Context:** PM2Lat的核心思想：不同kernel即使执行相同FLOPs也会有显著性能差异，因此需要基于kernel配置进行differentiation。对compute-intensive kernel使用FLOP-based interpolation，对memory-bound kernel使用operation count和memory access volume的线性回归。在A100上BF16 Linear layer误差10.3% vs NeuSight的70.5%。

**Confidence:** high

---

### 8. 新兴方法：NLTSP、TCL/Mamba、LLM-based Cost Model

**Claim:** NLTSP（Nested Loop Tree Structure Processing）直接从嵌套循环树结构提取特征，相比TenSet MLP在GPU上特征提取速度快41.4倍，搜索时间减少4.11倍。TCL引入Mamba架构作为cost model，在8个硬件平台中6个上取得最佳性能，模型大小仅0.35 MB。[^139^][^226^]

**Source:** "NLTSP: A cost model for tensor program tuning using nested loop trees," Journal of Systems Architecture, 2025; "TCL: Enabling Fast and Efficient Cross-Hardware Tensor Program Optimization via Continual Learning," arXiv:2604.12891, 2026.

**URL:** https://arxiv.org/html/2604.12891v1

**Date:** 2025-2026

**Excerpt:** "NLTSP achieves feature extraction speeds on CPU and GPU that are, on average, 97.9 times and 41.4 times faster, respectively, and can reduce the average search time for CPU and GPU workloads by 2.50 times and 4.11 times, respectively."

**Context:** 这些新兴方法代表了cost model架构的演进：从MLP到Transformer到Mamba，以及从手动特征工程到自动从程序结构提取特征的趋势。

**Confidence:** medium

---

## 技术方法论详解

### 模型架构对比

| 方法 | 模型类型 | 架构细节 | 输出目标 | 适用场景 |
|------|---------|---------|---------|---------|
| **NeuSight** | MLP | 8层隐藏层，512单元/层，ReLU | per-SM utilization | DNN训练/推理延迟预测 |
| **PipeWeave** | 轻量MLP | 3层(256/128/64)，ReLU+BN+Dropout | execution efficiency | Kernel级延迟预测 |
| **Habitat** | MLP | 8层隐藏层，1024单元/层，ReLU | 执行延迟（forward+backward） | 训练迭代时间预测 |
| **AutoTVM/Ansor** | XGBoost (GBDT) | 梯度提升树，ranking目标 | relative throughput ranking | Tensor program搜索 |
| **TenSet MLP** | MLP | 4层，RankLoss训练 | normalized latency | Tensor program预训练 |
| **TLP** | Transformer | 8-head self-attention + 2 residual blocks | normalized latency score | Tensor program tuning |
| **WaveTune** | Bilinear Model | wave-conditioned分段双线性：T = alpha*GL + beta*G + gamma*L + delta | kernel latency | Runtime kernel配置 |
| **PM2Lat** | 线性回归/插值 | FLOP-based rational correlation + interpolation | kernel latency | DNN模型延迟预测 |
| **TCL** | Mamba | 选择性状态空间模型 | latency ranking | 跨硬件tensor program优化 |

### 输入特征工程

**NeuSight的输入特征（per-tile，5维）[^3^]：**
1. 计算强度指标（operational intensity）
2. 内存带宽利用率（memory bandwidth utilization）
3. 计算资源利用率（compute utilization）
4. 目标数据大小与L2缓存比值
5. Tile size与GPU资源配置关系

这些特征在输入MLP前经过pre-processing：设备资源按SM数量归一化（per-SM peak FLOPS等），并计算各硬件资源的utilization率。

**PipeWeave的分析特征向量[^1^]：**
- Math Pipeline：GPU级别总操作数、总理论cycles；SM级别最大操作数、最大理论cycles
- MIO (Memory-I/O) Pipeline：GPU级别总内存需求、理论cycles（Global/L2）；SM级别最大内存需求、理论cycles（Global/L2/Shared）

**AutoTVM/Ansor的特征（164维+）[^226^]：**
- 计算特征：FLOPs、算术强度
- 内存访问特征：数据重用率、访存模式
- 循环变换特征：tiling sizes、unroll factors、vectorization
- 硬件参数：cache size、memory bandwidth、vector width、SM count等

### 训练方法与损失函数

**Ranking vs Regression：**
- **AutoTVM/Ansor**使用XGBoost进行**relative ranking**预测，目标是区分good/bad programs的相对顺序，而非预测绝对延迟[^229^]
- **TenSet MLP**使用**RankLoss**（pairwise/listwise）训练，输出normalized throughput或latency[^232^]
- **NeuSight**使用**sMAPE**（symmetric mean absolute percentage error）作为损失函数，直接预测utilization值[^3^]
- **PipeWeave**使用**MAPE**作为损失函数，预测execution efficiency[^1^]
- **TLP**使用**lambda rank loss + self-attention**，label为normalized latency（min_latency/latency）[^227^]

**WaveTune的least-squares fitting[^153^]：**
对每个<macro-config, wave> bucket独立做最小二乘拟合：
```
theta = argmin ||T_measured - (alpha*G*L + beta*G + gamma*L + delta)||^2
```
这本质上是一个分段线性回归问题，每个wave segment有独立的参数。

### 泛化策略

**跨硬件泛化：**
- **NeuSight**：利用publicly available的GPU规格（memory size、bandwidth、peak FLOPS、L2 cache），对新GPU（如Blackwell）仅需调整输入特征[^3^]
- **PipeWeave**：分析模型将task distribution映射为compact feature vector，MLP在不同硬件上训练后即可对新硬件做one-forward-pass预测[^1^]
- **TLP/MTL-TLP**：多任务学习（multi-task learning）利用source hardware的数据辅助target hardware的预测，仅需7%的target数据即可达到competitive performance[^227^]
- **TCL**：continual learning框架，通过知识蒸馏在新硬件上增量学习，避免从头收集大规模数据集[^226^]

**跨工作负载泛化：**
- **NeuSight**：按operator类型训练5个独立MLP（BMM、FC、Element-wise、Softmax、LayerNorm）[^3^]
- **PipeWeave**：per-kernel-category训练独立MLP（GEMM、Attention、BMSNorm等）[^1^]
- **WaveTune**：per-kernel-type构建dual-table，跨input sizes泛化[^153^]

---

## 架构适配性分析

### NVIDIA GPU架构支持

| 方法 | Volta (V100) | Ampere (A100) | Hopper (H100) | Ada (L4) | Blackwell (B200) | 注释 |
|------|-------------|---------------|---------------|----------|------------------|------|
| NeuSight | 训练集 | 训练集 | **unseen** | **unseen** | 可适配 | 未在H100/L4上训练仍表现良好 |
| PipeWeave | seen | seen | seen | - | **unseen** | 在11种GPU上评估，5种unseen |
| Habitat | 训练集 | **unseen** | - | - | - | A100上误差127% |
| WaveTune | - | - | yes | - | yes | 测试了H100、B200、MI355X等 |
| PM2Lat | - | A100 | - | L4 | RTX5070 | 覆盖mobile到server级GPU |

**Claim:** NeuSight展示了良好的新架构适配能力，因为它依赖于publicly available的硬件规格。对于Blackwell架构，仅需memory size、bandwidth和peak FLOPS即可进行预测，SM数量和L2 cache size在发布后会逐步公开。[^3^][^7^]

**Source:** Lee et al., arXiv:2407.13853.

**URL:** https://arxiv.org/html/2407.13853v2

**Date:** 2024-01-18

**Excerpt:** "For the Blackwell architecture, the latest NVIDIA GPU architecture announced, details on memory size, bandwidth, and peak FLOPs are already available. Information on the number of SMs and L2 cache size is unavailable, but based on prior years, should become available closer to the release date or shortly after the GPU is released."

**Context:** 这说明基于硬件feature + ML的方法具有较好的前向兼容性，只要核心硬件参数可获得即可适配新架构。

**Confidence:** high

---

### 精度支持（FP8/FP16/BF16/FP32）

**Claim:** PipeWeave在FP8、BF16/FP16和FP32三种精度上进行了全面评估，而NeuSight最初仅评估FP32。PM2Lat发现BF16的预测难度显著高于FP32，因为NVIDIA库在BF16下提供约100种MatMul算法组合（FP32仅约13种），导致NeuSight在BF16上A100的Linear layer误差达70.5%（PM2Lat仅10.3%）。[^1^][^50^]

**Source:** La et al., PM2Lat, arXiv:2603.00549.

**URL:** https://arxiv.org/html/2603.00549v1

**Date:** 2026-02-28

**Excerpt:** "In FP32, NVIDIA libraries offer about 13 combinations of algorithms and tile sizes for computing MatMul layers, whereas BF16 provides nearly 100. The higher number of combinations increases the contribution of unseen factors such as memory access patterns."

**Context:** 这表明ML方法的泛化能力受到底层库实现复杂度的显著影响，尤其是当算法选择空间增大时，基于简单特征（如FLOPs、wave count）的模型面临更大挑战。

**Confidence:** high

---

## ML方法的优缺点分析

### 优点

1. **捕获非线性交互**：MLP、Transformer等模型能够捕获硬件参数与kernel配置之间复杂的非线性交互，这是纯分析模型难以做到的。PipeWeave的MLP专门用于学习分析模型无法表征的高阶效应[^1^]。

2. **快速推理**：训练好的ML模型可以在微秒到毫秒级完成预测，适合在线决策场景。WaveTune仅需5-6 us，PM2Lat仅需0.045 ms[^91^][^50^]。

3. **跨硬件泛化**：通过将硬件参数作为输入特征，ML模型可以在unseen GPU上进行zero-shot或few-shot预测。NeuSight在未训练的H100上实现2.3%误差[^3^]。

4. **与搜索算法结合**：在tensor program tuning中，learned cost model（如XGBoost、MLP、Transformer）可以替代大量实际测量，将搜索时间从数天缩短到数小时[^232^][^227^]。

5. **可扩展性**：ML方法可以轻松集成新算子类型，只需收集数据并训练新模型即可。Habitat仅需为kernel-varying算子训练MLP，95%的算子自动用wave scaling处理[^168^]。

### 缺点

1. **需要大量训练数据**：ML模型的准确性高度依赖训练数据的覆盖范围和质量。NeuSight需要收集大量kernel profiling数据[^3^]；TenSet数据集包含5200万条测量记录[^232^]。对于新GPU或新算子，数据收集成本高昂。

2. **泛化能力有限**：
   - **维度外推**：Habitat在矩阵维度>1024时误差达38%[^7^]
   - **Unseen GPU**：Habitat在A100上误差达127%[^53^]
   - **新数据类型**：NeuSight在BF16上误差显著增大，因为训练时未充分考虑算法组合爆炸[^50^]
   - **Loss函数敏感性**：NeuSight使用sMAPE损失函数，对小延迟样本的误差过度敏感，导致模型偏向低延迟设备[^50^]

3. **Black-box特性**：纯ML方法缺乏可解释性，难以诊断预测错误的原因。PipeWeave通过分析-ML混合设计部分解决了这一问题[^1^]。

4. **推理开销**：虽然单次推理很快，但在大规模搜索中累积的推理开销不可忽视。XGBoost在WaveTune的实验中决策延迟达1822-2965 us，已超过多数kernel的执行时间[^91^]。

5. **与分析方法的权衡**：

| 方法类型 | 代表工作 | 精度 | 泛化性 | 解释性 | 数据需求 | 推理速度 |
|---------|---------|------|-------|-------|---------|---------|
| 纯分析模型 | Roofline, PALEO | 中 | 高 | 高 | 无 | 极快 |
| 纯ML模型 | Habitat, TenSet MLP | 中-高 | 低-中 | 低 | 高 | 快 |
| 分析+ML混合 | NeuSight, PipeWeave | 高 | 高 | 中 | 中 | 快 |
| 物理启发式 | WaveTune, PM2Lat | 高 | 高 | 高 | 低 | 极快 |

---

## 工具与资源

### 开源项目

1. **Habitat**: https://github.com/geoffxy/habitat - PyTorch-based runtime performance predictor
2. **TVM/AutoTVM**: https://github.com/apache/tvm - 包含XGBoost cost model和Ansor auto-scheduler
3. **TenSet Dataset**: https://github.com/tlc-pack/tenset - 大规模tensor program性能数据集
4. **TLP**: 集成于TVM/Ansor，基于Transformer的cost model

### 关键数据集

1. **TenSet** [^232^]: 120个网络，6个硬件平台，5200万+测量记录
2. **TenSet-TLP** [^227^]: 扩展版本，包含schedule primitive序列特征
3. **PipeWeave Dataset** [^1^]: ~1M样本，11种GPU，5类kernels，FP8/BF16/FP32

### 学术论文与代码

| 方法 | 论文 | 年份 | 会议/平台 |
|------|------|------|----------|
| Habitat | Yu et al., "A Runtime-Based Computational Performance Predictor" | 2021 | ATC |
| AutoTVM | Chen et al., "Learning to Optimize Tensor Programs" | 2018 | NeurIPS |
| Ansor | Zheng et al., "Generating High-Performance Tensor Programs" | 2020 | OSDI |
| TenSet | Zheng et al., "A Large-scale Program Performance Dataset" | 2021 | NeurIPS DBT |
| TLP | Zhai et al., "TLP: A Deep Learning-based Cost Model" | 2023 | ASPLOS |
| NeuSight | Lee et al., "Forecasting GPU Performance" | 2024 | arXiv |
| PipeWeave/SYNPERF | "Synergizing Analytical and Learning Models" | 2025 | arXiv |
| PM2Lat | La et al., "Highly Accurate and Generalized Prediction" | 2026 | arXiv |
| WaveTune | Zhang et al., "Wave-aware Bilinear Modeling" | 2026 | arXiv |
| NLTSP | "Nested Loop Tree Structure Processing" | 2025 | JSA |
| TCL | "Continual Learning for Cross-Hardware Optimization" | 2026 | arXiv |

---

## 关键引用列表

[^1^]: PipeWeave/SYNPERF, arXiv:2601.14910, 2025. https://arxiv.org/abs/2601.14910
[^2^]: NeuSight (v3), arXiv:2407.13853v3, 2024. https://arxiv.org/html/2407.13853v3
[^3^]: NeuSight (v2), arXiv:2407.13853v2, 2024. https://arxiv.org/html/2407.13853v2
[^4^]: NeuSight (PDF), arXiv:2407.13853, 2024. https://arxiv.org/pdf/2407.13853
[^7^]: Lee et al., "Data-driven Forecasting," arXiv:2407.13853v1, 2024. https://arxiv.org/html/2407.13853v1
[^12^]: PipeWeave PDF, arXiv:2601.14910, 2025. https://arxiv.org/pdf/2601.14910
[^49^]: WaveTune, arXiv:2604.10187, 2026. https://arxiv.org/abs/2604.10187
[^50^]: PM2Lat, arXiv:2603.00549v1, 2026. https://arxiv.org/html/2603.00549v1
[^51^]: WaveTune (HTML), arXiv:2604.10187, 2026. https://arxiv.org/html/2604.10187
[^52^]: Habitat, arXiv:2102.00527, 2021. https://arxiv.org/pdf/2102.00527
[^53^]: NeuSight v1, arXiv:2407.13853v1, 2024. https://arxiv.org/html/2407.13853v1?curius=2790
[^62^]: NeuSight Abstract, arXiv:2407.13853, 2024. https://arxiv.org/abs/2407.13853
[^91^]: WaveTune v1 HTML, arXiv:2604.10187v1, 2026. https://arxiv.org/html/2604.10187v1
[^92^]: PM2Lat Abstract, arXiv:2603.00549, 2026. https://arxiv.org/abs/2603.00549
[^139^]: NLTSP, Journal of Systems Architecture, 2025. https://www.sciencedirect.com/science/article/pii/S138376212400242X
[^153^]: WaveTune v1 (detail), arXiv:2604.10187v1, 2026. https://arxiv.org/html/2604.10187v1
[^160^]: ConvMeter, ACM, 2025. https://dl.acm.org/doi/fullHtml/10.1145/3673038.3673107
[^168^]: Habitat PDF, ATC 2021. https://www.cs.toronto.edu/ecosystem/papers/ATC_21/Habitat.pdf
[^226^]: TCL, arXiv:2604.12891v1, 2026. https://arxiv.org/html/2604.12891v1
[^227^]: TLP, ASPLOS 2023. https://ar5iv.labs.arxiv.org/html/2211.03578
[^229^]: Ansor, arXiv:2006.06762, 2020. https://arxiv.org/pdf/2006.06762
[^230^]: TLP PDF, arXiv:2211.03578, 2023. https://arxiv.org/pdf/2211.03578v2
[^232^]: TenSet, NeurIPS 2021 DBT. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf
[^233^]: TLP PDF (detail), arXiv:2211.03578, 2023. https://arxiv.org/pdf/2211.03578

---

## 待深入区域

1. **LLM-based Cost Model**: 最新研究开始探索使用Large Language Model（如Qwen2.5）直接作为cost model，通过fine-tuning LLM来理解tensor program语义并预测性能。这种方法可能利用LLM的泛化能力处理unseen program structures[^236^]。

2. **Multi-Task与Transfer Learning**: TLP的MTL-TLP和TCL的continual learning框架展示了跨硬件知识迁移的潜力，但如何在更多样化的硬件（如AMD GPU、TPU、NPU）之间高效迁移仍待研究[^227^][^226^]。

3. **Dynamic/Runtime Workload**: 当前ML方法主要针对static DNN topologies。对于MoE（Mixture-of-Experts）等动态计算流，需要预测token routing和expert activation，这是一个尚未充分解决的挑战[^50^]。

4. **Thermal and Power Modeling**: PM2Lat发现GPU thermal throttling对延迟有显著影响（T4/L4等passively cooled设备上误差32.6%），但现有ML方法很少显式建模thermal行为[^50^]。

5. **Blackwell架构支持**: 随着NVIDIA Blackwell架构的发布，需要验证现有ML方法对新架构（如FP4支持、第五代Tensor Core、第二代Transformer Engine）的适配能力[^3^]。

6. **Accuracy vs Efficiency Trade-off**: WaveTune展示了物理启发式模型可以在精度和效率上同时超越ML模型。未来研究需要探索更多hybrid方案，在保留ML的泛化能力的同时降低推理开销和数据需求[^91^]。

7. **Transformer-based方法的计算复杂度**: TLP等基于Transformer的cost model虽然精度高，但self-attention的O(n^2)复杂度限制了其在超大规模搜索空间中的应用。NLTSP和TCL（Mamba）通过更高效的架构部分解决了这一问题[^139^][^226^]。

---

*报告生成时间：2025年7月*
*本报告基于对20+篇学术论文和技术文档的深度调研编写*
