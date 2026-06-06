# [RL随笔] 小白学习verl AgentLoop 与 GSM8K Agentic RL Demo

**作者**: KevinMaster Student at ZJU, MLSys, AI Infra

**原文链接**: https://zhuanlan.zhihu.com/p/2042003898190356728

---

TL;DR：因实验室项目需要，本文浅浅分析了 verl 框架中 AgentLoop 的设计与实现（状态机驱动的多轮工具调用 + response_mask 梯度屏蔽 + 异步训练架构），并在 8×H20 上完成了一个最简单的 GSM8K + 计算器工具的 Agentic RL Demo，使用 Qwen2.5-7B-Instruct + GRPO 算法。文末列出了 8 个未来调研方向，涵盖数据流分析、框架对比、训推一致性等。
一、AgentLoop 概念与设计思路
1.1 什么是 AgentLoop

AgentLoop 是 verl 中实现 Agentic RL（智能体强化学习） 的核心接口，位于 verl/experimental/agent_loop/ 目录下。它负责管理 LLM 与环境/工具之间的多轮交互循环，并将交互轨迹转化为可训练的 RL 数据。

与传统单轮 RL（prompt → response → reward）不同，Agentic RL 的特点是：

Prompt → LLM 生成 → 调用工具 → 拼接工具结果 → LLM 继续生成 → ... → 最终回答 → Reward

AgentLoop 将这个”多轮交互 → 轨迹收集 → RL 训练”的流程完整封装。

1.2 架构总览
AgentLoop 在 verl RL 中的定位

在 verl 的 RL 训练流程中，AgentLoop 位于 Rollout 阶段，负责替代传统的单轮生成，实现 LLM 与工具/环境的多轮异步交互。整体定位如下：

verl RL 训练主循环
    ├── Rollout 阶段 ← AgentLoop 在这里
    │       ├── AgentLoopWorker（管理 batch 内每个样本的多轮交互）
    │       ├── ReactAgentLoop / ToolAgentLoop（单样本的 Agent 循环）
    │       └── AsyncServer（SGLang/vLLM 推理引擎）
    ├── Training 阶段（Actor/Critic 参数更新，GRPO/PPO）
    └── Weight Sync（将更新后的权重同步回推理引擎）

由于 Agent 需要通过工具调用与环境交互，为避免等待工具返回时 GPU 空闲，verl 采用了 Server-Client 分离架构 + asyncio 协程 的设计：

Server 端：AsyncServer 封装了 SGLang/vLLM 推理引擎，每个实例连接一个 DP group，对上层统一提供 generate 接口
Client 端：AgentLoopWorker 通过 LLMServerClient（推理网关）调用 Server，实现负载均衡和长尾请求优化
为什么 generate 接口基于 token 而非 text？ 因为 text ↔ token 的转换可能不可逆（例如 <think> 的 token 与 LLM 生成的不同）。训练阶段必须严格使用 LLM 推理产出的原始 token 来计算 advantage，否则会影响模型性能。基于 token 的 API 帮助 Client 维护工具调用文本与 LLM 返回 token 之间的正确映射。
组件层次
AgentLoopManager (分布式调度层)
    ├── AgentLoopWorker × N (Ray Worker，管理 batch 内每个样本)
    │       ├── AgentLoopBase.run() (单个样本的交互循环)
    │       ├── _compute_score() (奖励计算)
    │       ├── _compute_teacher_logprobs() (知识蒸馏)
    │       ├── _compute_multi_modal_inputs() (多模态处理)
    │       └── _postprocess() (合并为 DataProto batch)
    └── LLMServerClient (与 vLLM 推理服务通信)
1.3 核心组件
AgentLoopBase（抽象基类）
class AgentLoopBase(ABC):
    @abstractmethod
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        """单个样本的完整 agent 交互循环"""
注册机制：通过 @register("name") 装饰器注册，支持 hydra 动态实例化
输出：AgentLoopOutput 包含 input_ids、response_mask（区分 LLM 生成 vs 环境响应）、metrics、extra_fields
AgentLoopWorker（单 Worker 执行层）

AgentLoopWorker 是连接 AgentLoopManager（调度层） 和 AgentLoopBase（单样本交互） 的中间层，负责在单个 Ray Worker 上处理一个 batch 的样本。

核心职责：

工具加载：在 Worker 初始化时一次性加载所有 BaseTool 和 FunctionTool，后续每个样本复用
序列生成：generate_sequences(batch) 方法为 batch 中每个样本创建对应的 AgentLoopBase 实例并异步运行
后处理：将多个 AgentLoopOutput 合并为统一的 DataProto（含 input_ids、response_mask、attention_mask 等），供下游 Trainer 使用
奖励计算：调用 _compute_score() 计算每条轨迹的 reward（基于工具的 calc_reward 或外部 reward model）
蒸馏支持：可选地通过 teacher client 计算 teacher logprobs（用于在线策略蒸馏）
class AgentLoopWorker:
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        # 1. 为 batch 中每个样本实例化 AgentLoopBase（如 ToolAgentLoop）
        # 2. asyncio.gather 并行执行所有样本的 agent loop
        # 3. _postprocess：合并为 DataProto，pad 到统一长度
        # 4. 返回包含 prompts/responses/response_mask 的训练数据
ToolAgentLoop（ReAct 多轮工具调用）

状态机驱动：

PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED
    ↑___________________________|

关键特性：

response_mask 机制：LLM 生成 token mask=1（参与梯度），工具/环境响应 mask=0（不参与梯度）
多轮控制：max_user_turns、max_assistant_turns、max_parallel_calls
工具解析：支持 Hermes/Llama3/GPT-OSS 格式（通过 ToolParser）
Preemption：超长序列可被抢占截断，避免阻塞 batch
SingleTurnAgentLoop（单轮对话）
无工具调用，直接生成一次响应
支持多模态输入
作为基线或非交互场景使用
AgentLoopManager（分布式调度）
Ray 分布式：将 batch 切分到 N 个 AgentLoopWorker，round-robin 调度到不同节点
异步并行：asyncio.gather 并行执行所有 worker
性能指标：追踪 slowest sample、generate/tool_calls/compute_score 耗时、preemption 次数
1.4 工具系统
类型	基类	特点	适用场景
BaseTool	BaseTool	有状态，create/execute/calc_reward/release 生命周期	模拟器环境、代码沙箱
FunctionTool	@function_tool 装饰器	无状态，自动推导 OpenAI schema	计算器、搜索等简单工具

工具 Schema 与 OpenAI function calling 格式完全兼容，ToolResponse 支持 text/image/video 多模态返回。

1.5 与 Fully Async Trainer 的集成

AgentLoop 同时支持异步训练框架（FullyAsyncRollouter）和同步训练框架（main_ppo_sync.py 中的 AgentLoopWorkerTQ），但异步模式是推荐的使用方式。核心原因：

多轮工具调用天然是异步交互（生成→调工具→继续生成）
Agentic 场景每个样本耗时方差极大，异步模式避免慢样本阻塞全部
异步训练架构
┌─────────────────────────────────────────────────────────────┐
│                    FullyAsyncTaskRunner                       │
│                                                              │
│  ┌─────────────────────┐       ┌──────────────────────────┐ │
│  │  FullyAsyncRollouter│       │   FullyAsyncTrainer      │ │
│  │   (GPU 0-3)         │       │     (GPU 4-7)            │ │
│  │                     │       │                          │ │
│  │  _feed_samples()    │       │  fit() 主循环            │ │
│  │       ↓             │       │    ↓                     │ │
│  │  pending_queue      │       │  从 MQ 取样本            │ │
│  │       ↓             │       │    ↓                     │ │
│  │  _processor_worker()│       │  compute_advantages()    │ │
│  │       ↓             │       │    ↓                     │ │
│  │  AgentLoopManager   │       │  update_actor()          │ │
│  │       ↓             │       │    ↓                     │ │
│  │  MessageQueue ──────┼──MQ──→│  每N步 sync weights ──┐  │ │
│  │       ↑             │       │                       │  │ │
│  │       │             │←──────┼── reset_staleness() ←─┘  │ │
│  └─────────────────────┘       └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
Staleness 控制机制

异步训练中，Rollout 端生成的样本可能使用的是旧版策略，通过 staleness 控制 on-policy / off-policy 的程度：

参数	含义
staleness_threshold	样本策略版本与当前版本差距阈值，超过则丢弃
trigger_parameter_sync_step	每隔多少 training step 同步权重到 rollout
Preempt + 自动恢复

FullyAsyncLLMServerClient 在权重同步时，正在推理的请求会被 preempt。客户端自动将已生成的 token 作为新 prompt 继续生成，对上层 AgentLoop 透明。

二、GSM8K Agentic RL Demo 实现

2.1 整体设计

在 8×H20（144GB/卡）上，使用 Qwen2.5-7B-Instruct + 计算器工具 + GRPO 算法，通过 AgentLoop 实现多轮工具调用的 Agentic RL 训练。

GPU 分配：

GPU 0-3: vLLM Rollout（推理生成，TP=4）
GPU 4-7: FSDP2 Training（策略梯度更新）

训练流程：

Prompt → AgentLoop 多轮交互（LLM生成 → 调用计算器 → 拼接结果 → 继续生成）
→ 收集完整轨迹 → response_mask 标记 → GRPO 更新
2.2 实现步骤
步骤 1：环境安装
cd /root/rl_research/verl
uv venv --python 3.12
source .venv/bin/activate

# 按照 verl 官方文档安装
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
uv pip install --no-deps -e .
NOTE：scripts/install_vllm_sglang_mcore.sh 脚本内部使用的是 pip install，在 uv 虚拟环境下需要将其中的 pip install 替换为 uv pip install，否则可能安装到系统环境或报错。
步骤 2：下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
步骤 3：准备数据

使用 verl 自带的数据预处理脚本，将 GSM8K 转成带 agent_name: "tool_agent" 和 tools_kwargs 字段的 parquet。

python examples/data_preprocess/gsm8k_tool_agent_loop.py --local_save_dir ~/data/gsm8k

数据中的关键字段：

agent_name: "tool_agent" → 告诉 AgentLoop 使用 ToolAgentLoop
tools_kwargs → 工具初始化参数
reward_model.ground_truth → 标准答案，用于 DAPO reward 计算
步骤 4：实现工具

verl 的数据预处理脚本引用了 calc_gsm8k_reward 工具名，但仓库中没有提供实现。我们使用 @function_tool 装饰器实现了一个简单的计算器：

# my_experiment/gsm8k_tools.py
@function_tool("calc_gsm8k_reward")
def calc_gsm8k_reward(expression: str) -> str:
    """Evaluate a mathematical expression and return the numerical result."""
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

工具名必须与数据中的引用一致，因为：

System Prompt 告诉模型用 calc_gsm8k_reward
模型生成 tool_call 时写这个名字
AgentLoop 通过名字查找并调用对应工具

注意：这个工具只是辅助计算，不是 reward 函数。真正的 reward 由 DAPO reward manager 在轨迹结束后比较最终答案与 ground truth 得到。

步骤 5：编写启动脚本

基于 dapo_7b_async_retool.sh 模板，适配 8×H20 环境。关键配置：

# 异步训练
actor_rollout_ref.rollout.mode=async

# 启用 AgentLoop 多轮工具调用
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.max_user_turns=4
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
actor_rollout_ref.rollout.multi_turn.function_tool_path=$FUNCTION_TOOL_PATH
actor_rollout_ref.rollout.multi_turn.format=hermes

# 异步控制
async_training.staleness_threshold=0.5
async_training.trigger_parameter_sync_step=4

完整脚本见 my_experiment/run_gsm8k_agentic_rl.sh。

2.3 踩坑记录
坑 1：gen_batch_size 必须为 1
AssertionError: gen_batch_size must be one

原因：Fully Async 模式下每个 rollout step 只处理单个 prompt，然后对它生成 n_resp_per_prompt 个回复。

修复：gen_prompt_bsz=1

深层原因：gen_batch_size=1 不代表同一时刻只有一个 prompt 在推理。Fully Async 通过 asyncio 并发控制，同时有 max_concurrent_samples（默认 16）个 prompt 在并发处理（见 fully_async_rollouter.py:904），只是每个任务的调度粒度是单个 prompt。这样设计的原因：

避免慢样本阻塞：Agentic 场景每个样本的工具调用轮数差异大（1~8 轮），按 batch 处理必须等最慢的那个完成
流式提交：每个样本完成后立即通过 MQ 发给 Trainer（fully_async_rollouter.py:933），不需要凑满 batch
GPU 不受影响：vLLM 内部的 continuous batching 自动合并并发请求，GPU 利用率不受上层调度粒度限制
坑 2：NumPy 版本不兼容
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.4.

原因：vLLM worker 进程依赖 numba，而 numba 不兼容 NumPy 2.4。

修复：uv pip install "numpy<2.3"

坑 3：工具调用解析失败（预期行为）
ERROR: Failed to decode tool call: 'name'
ERROR: Failed to decode tool call: Expecting value: line 2 column 1

原因：训练初期模型还没学会正确的 Hermes 格式 tool call，输出的 JSON 格式不对。这是预期行为，随着训练进行会逐渐减少。ToolAgentLoop 会 gracefully 处理，解析失败时不调用工具，模型继续生成。

坑 4：**kwargs 不被 @function_tool 支持
ValueError: @function_tool 'calc_gsm8k_reward' declares variadic parameter(s) ['kwargs'],
which can't be expressed in an OpenAI tool schema.

原因：@function_tool 从函数签名自动推导 OpenAI tool schema，**kwargs 无法表达为 JSON Schema。

修复：去掉 **kwargs，保持显式参数签名。模型传的多余参数由 ToolAgentLoop 层面过滤。

坑 5：权重同步 OOM
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.

原因：gpu_memory_utilization=0.85 太高，vLLM 占了 132 / 140 GiB，权重同步的 checkpoint engine 需要分配 2GB buffer（默认 update_weights_bucket_megabytes=2048）时 OOM。

修复：

gpu_memory_utilization 从 0.85 降到 0.75
update_weights_bucket_megabytes 从 2048 降到 512（对性能影响可忽略，只是传输分块更小）
2.4 关键参数解释
Training Step 的计算
total_train_steps = total_rollout_steps / (ppo_mini_batch_size × trigger_parameter_sync_step)

例如要训 50 步：total_rollout_steps = 50 × 16 × 4 = 3200

Reward 机制
组件	作用	时机
calc_gsm8k_reward（工具）	辅助计算器，帮模型验证中间步骤	Rollout 过程中
DAPO Reward Manager	提取最终答案 #### <answer> 与 ground truth 比较	轨迹结束后
三、训练结果
3.1 实验配置
项目	值
模型	Qwen/Qwen2.5-7B-Instruct
数据集	GSM8K（7473 训练 / 1319 测试）
算法	GRPO（无 Critic）
工具	计算器（@function_tool）
GPU	8×H20（144GB/卡）
Rollout GPU	4（vLLM TP=4）
Training GPU	4（FSDP2）
学习率	1e-6
staleness_threshold	0.5
3.2 训练指标（完整 50 步）
注：验证集指标（val-core、num_turns）每 10 步评估一次，Step 0 为训练前的基线评估；训练指标（score、kl、pg_loss、clipfrac）每步记录。
指标	Step 0/1	Step 10	Step 20	Step 30	Step 40	Step 50
val-core/gsm8k/acc/mean@1	0.626	0.904	0.904	0.913	0.909	0.920
critic/score/mean	0.697	0.879	0.919	0.960	0.930	0.969
actor/ppo_kl	0.001	0.002	0.0013	0.0092	0.0008	0.0016
actor/pg_loss	-0.055	0.024	0.011	0.013	0.015	0.045
actor/pg_clipfrac	0.001	0.001	0.0003	0.0007	0.0004	0.0005
val-aux/num_turns/mean	4.36	4.24	4.10	3.97	3.99	3.81
3.3 最终结果

训练共 50 步，总耗时约 72 分钟（8×H20），throughput 约 1089 tokens/s。

项目	初始值	最终值（Step 50）	变化
验证集准确率	62.6%	92.0%	+29.4%
训练集 reward	0.697	0.969	+0.272
平均对话轮数	-	3.81	逐步下降，模型学会更高效地使用工具
KL 散度	0.004	0.0016	策略更新稳定，未发生 KL 爆炸
Clip 比例	0.003	0.0005	极低，说明 GRPO 更新幅度受控
3.4 结论
训练效果显著：GSM8K 验证集准确率从 62.6% 提升至 92.0%（+29.4%）
训练完全稳定：KL 散度始终在 0.001~0.009 范围内波动，策略 loss 和 clip 比例曲线均健康
AgentLoop 正常工作：模型学会了多轮工具调用，平均对话轮数从 4.24 逐步下降到 3.81，说明模型学会了更高效地使用计算器
全流程跑通：从数据预处理 → 工具实现 → 异步训练 → 评估的完整 Agentic RL 链路，总耗时约 72 分钟
四、未来调研方向
4.1 详细分析 AgentLoop 的数据流

从 prompt 进入 _feed_samples() 到最终 training step 完成的完整 tensor 流转路径，包括：

prompt → AgentLoop 多轮交互 → AgentLoopOutput（含 response_mask）
_postprocess() 合并为 DataProto → MQ 传输 → Trainer 取样本
compute_advantages()（GRPO group-relative）→ update_actor() 梯度更新
重点关注 response_mask 在 loss 计算中如何屏蔽工具响应 token
4.2 对比其他 RL 框架的 Agentic RL 实现
框架	待调研点
Slime	多轮交互的抽象方式、工具调用集成、与 verl AgentLoop 的设计差异
AReaL	异步训练架构、search agent 实现（react_agent.py）、LLM-as-Judge reward
Miles	多智能体 RL 场景、环境交互接口设计

重点对比：状态机 vs 其他交互抽象、response_mask 机制的异同、异步训练支持程度。

4.3 多模态 RL 的实现流程和比较
verl 中 _compute_multi_modal_inputs() 的实现（图像/音频/视频的处理链路）
SingleTurnAgentLoop 对多模态的支持 vs ToolAgentLoop 中多模态工具返回（ToolResponse 支持 image/video）
对比不同框架对多模态 RL 的支持程度
4.4 BaseTool 有状态工具的深入调研与实践

当前 Demo 使用的是 @function_tool（无状态），verl 还支持 BaseTool 有状态工具：

生命周期管理：create() → execute() → calc_reward() → release()
适用场景：代码沙箱（执行代码并根据运行结果给 reward）、模拟器环境（机器人控制、游戏等）
待实践：用 BaseTool 实现一个代码执行沙箱工具，在 MATH/HumanEval 等任务上验证效果
4.5 Reward 设计策略调研

当前使用简单的答案匹配 0/1 reward，可调研更丰富的 reward 方案：

Process Reward Model (PRM)：对推理过程的每一步给分，而非只看最终答案
Outcome Reward Model (ORM)：训练专用的 reward 模型评估结果质量
LLM-as-Judge：用另一个 LLM 评估回答质量（AReaL 中已有实现 calc_reward_with_llm_judge）
分析不同 reward 方案在 Agentic RL 多轮交互场景下的效果差异和适用边界
4.6 Staleness 与 Off-policy 程度对训练效果的影响

当前实验使用 staleness_threshold=0.5、trigger_parameter_sync_step=4，可设计消融实验：

不同 staleness_threshold（0.1 / 0.3 / 0.5 / 1.0）对收敛速度和最终准确率的影响
不同 trigger_parameter_sync_step（1 / 2 / 4 / 8）对 throughput 和训练效果的权衡
分析 dropped_stale_samples 比例与训练稳定性的关系
4.7 RL 过程中训推一致性的问题定位和分析

RL 训练过程中 Rollout（推理）和 Training（训练）之间可能存在数值不一致（Training-Inference Mismatch），即使使用相同权重，两端对同一 token 序列计算出的 log prob 也会有差异。这会导致 PPO/GRPO 的 importance sampling ratio 出现偏差，严重时引发训练崩溃。

问题来源
浮点非结合性：不同 CUDA kernel 的 reduction 顺序不同（如 batch size 变化导致 split-reduction 策略不同），浮点加法非结合性导致数值差异
训推引擎 kernel 差异：Rollout 端（vLLM）使用针对小 batch 优化的推理 kernel，Training 端（FSDP2/Megatron）使用针对大 batch 优化的训练 kernel，两者 forward pass 结果存在微小差异
MoE 模型放大效应：MoE 模型的 expert routing 对输入极度敏感，微小的数值差异可能导致选择不同的 expert，K3 KL 从 dense 模型的 10⁻⁵~10⁻³ 放大到 10⁻³~10⁻¹
异步架构下的权重版本差异：Rollout 使用的模型权重可能落后于 Training 端，staleness 进一步加剧 off-policy 程度
Preempt 导致的轨迹拼接问题：权重同步时正在推理的请求被 preempt，恢复后继续生成的 token 使用了新权重，但 log prob 可能按旧权重计算
现有解决方案

方案一：Truly On Policy（训推 kernel 对齐）

对齐训推两端的所有算子后端，使 log prob 逐位一致（bitwise identical），KL = 0。

Miles 框架：对齐 SGLang + FSDP/Megatron 的 RMSNorm、Matmul、FlashAttention-3、DeepGEMM 等算子
vLLM + TorchTitan：审计 forward pass 中每个 kernel 调用，使用 batch-invariant kernel 实现跨框架逐位一致
代价：需要侵入性修改训练引擎，性能有损失（vLLM+TorchTitan 报告约 2.4x 减速），且目前仅对 dense 模型有效

方案二：算法层面纠正（不对齐 kernel，纠正 loss）

将 Rollout 端的 log prob 视为 behavior policy 的真实值，通过算法纠正 off-policy 误差：

TIS（Truncated Importance Sampling）：对 importance sampling ratio 做截断，当训推 log prob 差异过大的 token 被截断，避免极端 ratio 破坏梯度
MIS（Masked Importance Sampling）：对差异超过阈值的 token 直接 mask 掉，不参与 loss 计算
IcePop：蚂蚁团队提出的 token 级 discrepancy masking 算法，专门解决 MoE 模型的 RL 训练稳定性问题，在 GLM 4.5⁄4.6⁄4.7 等前沿模型的后训练中验证了工业级稳定性
By-Passing Old Log-Prob：直接使用 Rollout 端的 log prob 作为 PPO importance sampling 的分母（而非 Training 端 recompute 的值），恢复数学正确性
待调研重点
分析 verl 中 vLLM Rollout 和 FSDP2 Training 之间的 K3 KL 散度量级
调研 verl 是否已实现 TIS/MIS 等纠正机制，或需要自行集成
对比 Truly On Policy（精确但慢）vs 算法纠正（快但近似）在不同场景下的权衡
特别关注 Agentic RL 多轮交互场景下 mismatch 的累积效应
4.8 Long Horizon Task 的 RL 策略与效率优化

Long horizon task 指需要大量交互步骤才能完成的任务（如多步网页操作、复杂代码生成、长链路搜索推理等），其轨迹长度远超 GSM8K 这类短 horizon 任务。需要调研：

Long horizon 的定义与典型 benchmark：WebArena、SWE-bench、深度研究（Deep Research）等任务的轨迹长度特征
稀疏 reward 问题：长轨迹中只有最终步有 reward 信号，中间步缺乏反馈，导致 credit assignment 困难；PRM、中间 reward shaping 等解决方案
效率瓶颈：长序列的 KV cache 显存占用、多轮工具调用的延迟累积、vLLM continuous batching 在超长序列下的调度效率
算法层面的优化策略：hierarchical RL（分层决策）、curriculum learning（从短 horizon 逐步增加）、trajectory truncation 与 partial reward、Monte Carlo Tree Search (MCTS) 等
工程层面的效率优化：序列并行、KV cache offload、speculative decoding 在长轨迹 rollout 中的应用、preemption 策略对长序列的影响

注

本文绝大部分由Claude Opus 4.7分析生成，如有错误请批评指正
