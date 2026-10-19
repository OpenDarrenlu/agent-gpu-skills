---
name: verl-skill
description: >
  verl (HybridFlow, verl-project/verl) RL post-training framework skill for LLM/VLM reinforcement
  learning. Use for verl, HybridFlow, PPO/GRPO/DAPO/RLOO/REINFORCE++/GSPO/CISPO, RLHF/RLVR, main_ppo,
  V1 trainer (sync/colocate_async/separate_async), TransferQueue, single controller, WorkerGroup,
  DataProto, ActorRolloutRefWorker, model engine (FSDP/FSDP2/Megatron/VeOmni/TorchTitan),
  vLLM/SGLang/TRT-LLM rollout, hybrid-engine weight sync, checkpoint engine, agent loop, multi-turn
  tool RL, reward function/reward loop, on-policy distillation, LoRA RL, rollout correction (TIS/RS),
  train-inference mismatch, full determinism, router replay, Hydra config, RL OOM/throughput, nsys
  profiling, model_merger. 中文触发词：verl、HybridFlow、强化学习后训练、RL 训练框架、PPO/GRPO
  训练、RLHF、奖励函数、多轮工具调用 RL、Agentic RL、训推一致、训推不一致、rollout 权重同步、异步 RL、
  在线蒸馏。
triggers:
  - "verl"
  - "HybridFlow"
  - "GRPO"
  - "PPO"
  - "DAPO"
  - "RLHF"
  - "RLVR"
  - "main_ppo"
  - "RayPPOTrainer"
  - "DataProto"
  - "WorkerGroup"
  - "TransferQueue"
  - "agent loop"
  - "reward function"
  - "rollout correction"
  - "强化学习"
  - "后训练"
  - "训推一致"
  - "奖励函数"
---

# verl (HybridFlow) RL Post-Training

verl 是 ByteDance Seed 开源、现由 `verl-project` 维护的 LLM/VLM 强化学习后训练框架
（HybridFlow 论文实现）。本 skill 提供源码地图、扩展点、配置语义、调试与性能排查流程。

## 边界与转交

| 问题落点 | 处理方 |
|---|---|
| RL dataflow、trainer、worker、model engine、reward、agent loop、Hydra 配置、RL 侧 OOM/吞吐 | **本 skill** |
| SGLang server 内部（scheduler、KV cache、attention backend） | `sglang-skill` |
| 权重同步 / NCCL 集合通信 hang、IB/RoCE、NIXL/Mooncake 传输 | `nccl-skill` / `gpu-communication-libraries` |
| 已定位到单个 kernel 慢（attention、GEMM、fused CE） | `gpu-performance-router` → 具体 kernel skill |
| 整机 nsys 时间线解读 | `nsys-profile-analysis` / `perf-nsight-systems` |
| 纯 serving 部署 / 容量规划（不涉及训练） | `llm-serving-router` |

## Source Code Locations

verl 源码位于此 skill 安装目录下的 `repos/verl/`。实际路径取决于所用工具:
- Cursor: `~/.cursor/skills/verl-skill/repos/verl/`
- Claude Code: `~/.claude/skills/verl-skill/repos/verl/`
- Codex: `~/.codex/skills/verl-skill/repos/verl/`

**VERL_REPO**: 下文用 `VERL_REPO` 指代上述路径，**替换为实际路径**。
若路径不存在，在 agent-gpu-skills 项目根目录运行 `bash update-repos.sh verl`。

Sparse checkout 内容：`verl/`、`examples/`、`docs/`、`scripts/`、`tests/` 与根目录文件
（`pyproject.toml`、`setup.py`、`requirements*.txt`、`uv.lock`、`manage_envs.py`、上游 `AGENTS.md`）。
`recipe/` 在上游是指向 `verl-project/verl-recipe` 的 **git submodule**，默认不拉取；需要 DAPO/PRIME/
SPIN 等 recipe 源码时运行 `git -C VERL_REPO submodule update --init recipe`。

本 skill 编写时的上游快照：`0.10.0.dev` @ `263653e`（2026-10-19）。verl 迭代很快，
**引用行号前先用 rg 重新定位**。

## 架构速览

HybridFlow = **single-controller**（driver 编排 RL dataflow）+ **multi-controller**（每个
WorkerGroup 内部是 torch.distributed SPMD）。

```
python -m verl.trainer.main_ppo (Hydra: verl/trainer/config/ppo_trainer.yaml)
  └─ ray.init → TaskRunnerV1 (Ray actor, driver)           # trainer.use_v1=true (默认)
       ├─ PPOTrainer{Sync|ColocateAsync|SeparateAsync}.init()
       │    ├─ RayWorkerGroup(ActorRolloutRefWorker)         # actor + ref (+ rollout adapter) 同进程
       │    ├─ RayWorkerGroup(TrainingWorker, value_model)   # critic（GAE 时）
       │    ├─ LLMServerManager → RolloutReplica × N          # vLLM/SGLang/TRT-LLM HTTP server
       │    ├─ CheckpointEngineManager                        # trainer → rollout 权重同步
       │    ├─ RewardLoopManager                              # 规则 reward / GenRM / 远端 reward
       │    └─ MultiTeacherModelManager (可选, 在线蒸馏)
       ├─ AgentLoopManagerTQ.generate_sequences → 轨迹写入 TransferQueue
       └─ trainer.fit():  每 step
            replay_buffer.sample → reward → balance_batch → old_log_prob → ref_log_prob
            → values → advantage(driver 上计算) → update_critic → update_actor
            → on_step_end: checkpoint_manager.update_weights (sleep/wake rollout)
```

- **V1 trainer**（默认）：`verl/trainer/ppo/v1/`，以 TransferQueue 为数据面、ReplayBuffer 控制
  off-policy 程度（`trainer.v1.sampler.max_off_policy_threshold`）。三种模式由
  `trainer.v1.trainer_mode` 选择：`sync`（共置、无 partial rollout）、`colocate_async`（共置 +
  partial rollout，step 间 abort/resume）、`separate_async`（trainer/rollout 分离，
  `parameter_sync_step` 控制同步频率，可借出空闲 trainer GPU 做 rollout）。
- **V0 trainer**（legacy，已 deprecated）：`trainer.use_v1=false` → `main_ppo_v0.py` +
  `ppo/ray_trainer.py::RayPPOTrainer`。仍是 **bitwise full determinism 的必需路径**（见训推一致章节）。
- **Rollout 模式**（`verl/workers/rollout/replica.py::RolloutMode`）：`HYBRID`（与训练引擎同进程、
  共享 GPU、通过权重同步切换）、`COLOCATED`（同 placement group 不同进程，如 GenRM）、
  `STANDALONE`（独立 GPU，off-policy/分离式）。

## Source Map

```
VERL_REPO/verl/
├── protocol.py                       # DataProto / DataProtoFuture（batch: TensorDict, non_tensor_batch, meta_info）
├── base_config.py                    # BaseConfig（dataclass 配置基类）
├── single_controller/
│   ├── base/decorator.py             # @register, Dispatch(ONE_TO_ALL/ALL_TO_ALL/DP_COMPUTE*/...), Execute,
│   │                                 # make_nd_compute_dataproto_dispatch_fn(mesh_name)
│   ├── base/worker.py, worker_group.py
│   └── ray/base.py                   # RayResourcePool, ResourcePoolManager, RayClassWithInitArgs,
│                                     # RayWorkerGroup(spawn/fuse), create_colocated_worker_cls
├── trainer/
│   ├── main_ppo.py                   # Hydra 入口, run_ppo, TaskRunnerV1
│   ├── main_ppo_v0.py                # legacy TaskRunner（deprecated）
│   ├── ppo/
│   │   ├── core_algos.py             # 优势估计器 / policy loss / KL / agg_loss 注册表
│   │   ├── ray_trainer.py            # legacy RayPPOTrainer + compute_advantage/apply_kl_penalty/Role
│   │   ├── v1/                       # trainer_base.py(PPOTrainer, register_trainer), trainer_sync.py,
│   │   │                             # trainer_colocate_async.py, trainer_separate_async.py,
│   │   │                             # replay_buffer.py, agent_loop_tq.py
│   │   ├── utils.py                  # Role 枚举, need_critic/need_reference_policy, create_rl_dataset
│   │   ├── reward.py, metric_utils.py, rollout_corr_helper.py, score_centering.py
│   │   └── checkpoint_callback.py
│   ├── sft_trainer.py                # torchrun 多控制器 SFT（model engine）
│   ├── sft_trainer_ray.py            # [EXPERIMENTAL] 单控制器 SFT
│   ├── distillation/                 # 在线蒸馏 loss（losses.py, fsdp/, megatron/）
│   ├── main_generation_server.py     # 独立生成
│   ├── main_eval.py                  # 离线打分评测
│   └── config/                       # Hydra 配置（见“配置”章节）
├── workers/
│   ├── engine_workers.py             # TrainingWorker, ActorRolloutRefWorker（update_actor,
│   │                                 # compute_log_prob, compute_ref_log_prob, update_weights）
│   ├── engine/
│   │   ├── base.py                   # BaseEngine 接口 + EngineRegistry(model_type, backend, device)
│   │   ├── fsdp/                     # FSDPEngine(WithLMHead/WithValueHead), fsdp_turbo_impl.py
│   │   ├── megatron/                 # MegatronEngine(WithLMHead/WithValueHead), delta_export.py
│   │   ├── veomni/  torchtitan/  automodel/  mindspeed/(NPU)
│   ├── rollout/
│   │   ├── base.py                   # BaseRollout, _ROLLOUT_REGISTRY: (vllm|sglang|trtllm, async) → ServerAdapter
│   │   ├── replica.py                # RolloutMode, RolloutReplica, RolloutReplicaRegistry
│   │   ├── llm_server.py             # LLMServerClient, FullyAsyncLLMServerClient, LLMServerManager
│   │   ├── router.py                 # GlobalRequestLoadBalancer（请求路由 / sticky）
│   │   ├── vllm_rollout/             # vllm_async_server.py(vLLMHttpServer, vLLMReplica), vllm_rollout.py,
│   │   │                             # bucketed_weight_transfer.py, delta_weight_transfer.py, vllm_pd_replica.py
│   │   ├── sglang_rollout/           # async_sglang_server.py(SGLangHttpServer, SGLangReplica),
│   │   │                             # sglang_rollout.py, http_server_engine.py, sglang_pd_replica.py
│   │   └── trtllm_rollout/  hf_rollout.py  naive/
│   ├── config/                       # actor.py critic.py rollout.py engine.py model.py optimizer.py reward.py ...
│   ├── utils/losses.py               # ppo_loss / value_loss / sft_loss（engine 侧 loss 入口）
│   ├── utils/padding.py              # left/right padding ↔ no-padding(nested) 转换
│   └── reward_manager/               # legacy reward manager: naive / batch / dapo / prime
├── checkpoint_engine/                # 权重同步后端 backend 值：naive, nccl, nixl, mooncake, kimi_ckpt_engine,
│                                     # delta_sharded（昇腾上 HCCL 实现同样注册为 "nccl"）；README.md 有对比表与 benchmark
├── experimental/
│   ├── agent_loop/                   # AgentLoopBase, @register, AgentLoopWorker, AgentLoopManager,
│   │                                 # single_turn_agent_loop.py, tool_agent_loop.py, tool_parser.py
│   ├── reward_loop/                  # RewardLoopManager, reward_manager/(naive,dapo,gdpo,limited,remote), router/
│   ├── teacher_loop/                 # 蒸馏 teacher 服务
│   ├── fully_async_policy/           # Meituan 全异步 trainer/rollouter 方案
│   ├── one_step_off_policy/          # one-step-off 异步方案
│   └── separation/                   # trainer/rollout 分离实验
├── tools/                            # BaseTool, tool_registry.py, function_tool.py, schemas.py
├── models/
│   ├── transformers/monkey_patch.py  # apply_monkey_patch: Ulysses SP attention、VLM 切片、prefix grouper
│   ├── transformers/*.py             # 各模型补丁（qwen2/qwen2_vl/qwen3_vl/qwen3_5/glm4v/kimi_vl/llama ...）
│   └── mcore/                        # Megatron-Core: bridge.py, registry.py, model_forward*.py, mtp_patch.py
├── model_merger/                     # python -m verl.model_merger merge --backend fsdp|megatron|veomni
├── plugin/platform/                  # CUDA / ROCm / NPU 平台抽象
└── utils/
    ├── seqlen_balancing.py           # karmarkar_karp, get_seqlen_balanced_partitions, rearrange_micro_batches
    ├── ulysses.py                    # Ulysses SP all-to-all（gather_seq_scatter_heads 等）
    ├── dynamic_cp_scheduler.py       # Megatron dynamic context parallel
    ├── kernel/                       # linear_cross_entropy.py（fused linear+CE / Liger）, fp8_kernel.py
    ├── profiler/                     # ProfilerConfig, DistProfiler, nvtx/torch/mstx/torch_memory profiler
    ├── dataset/rl_dataset.py         # RLHFDataset, collate_fn
    ├── reward_score/__init__.py      # default_compute_score: data_source → 打分器
    ├── debug/metrics.py              # training/rollout_probs_diff_* 训推概率差指标
    ├── checkpoint/                   # fsdp/megatron checkpoint manager
    ├── vllm/  sglang/  trtllm/       # 推理侧 patch 与 FP8/FP4 量化工具
    ├── qat/  modelopt/  fp8_utils.py # QAT / 低精度
    ├── config.py                     # omega_conf_to_dataclass, validate_config
    └── tracking.py                   # trainer.logger 后端：console/wandb/swanlab/mlflow/tensorboard/clearml/file ...
```

其余目录：`VERL_REPO/examples/`（按算法分的启动脚本，见下）、`VERL_REPO/docs/`、
`VERL_REPO/scripts/`、`VERL_REPO/tests/`。

## 扩展点速查

| 想做的事 | 入口 | 选择方式 |
|---|---|---|
| 新优势估计器 | `core_algos.py::register_adv_est` | `algorithm.adv_estimator=<name>` |
| 新 policy loss | `core_algos.py::register_policy_loss` | `actor_rollout_ref.actor.policy_loss.loss_mode=<name>` |
| 自定义 reward 函数 | 任意 `.py` 中的 `compute_score(data_source, solution_str, ground_truth, extra_info, **kw)` | `reward.custom_reward_function.path=... reward.custom_reward_function.name=compute_score` |
| 自定义 reward manager | `experimental/reward_loop/reward_manager/registry.py::register` | `reward.reward_manager.name=<name>`（`source=register`），或 `reward.reward_manager.source=importlib` + `reward.reward_manager.module.{path,name}` |
| 自定义 agent loop | `experimental/agent_loop/agent_loop.py::register("<name>")` 继承 `AgentLoopBase.run` | 数据集字段 `agent_name`，缺省 `rollout.agent.default_agent_loop` |
| 新工具 | 继承 `tools/base_tool.py::BaseTool`（create/execute/calc_reward/release）或 `function_tool` 装饰器 | `rollout.multi_turn.tool_config_path` / `function_tool_path` |
| 工具调用格式 | `tool_parser.py::ToolParser.register` | `rollout.multi_turn.format`（hermes, gpt-oss, qwen3_coder, glm, seed, minimax, kimi, deepseek_v4, gemma4） |
| 新训练后端 | `workers/engine/base.py::EngineRegistry.register(model_type, backend, device)` | `model_engine=dp|megatron|veomni|torchtitan`，actor 的 `strategy` |
| 新推理后端 | `workers/rollout/base.py::_ROLLOUT_REGISTRY` + `replica.py::RolloutReplicaRegistry` | `actor_rollout_ref.rollout.name=vllm|sglang|trtllm` |
| 权重同步后端 | `checkpoint_engine/base.py::CheckpointEngineRegistry.register` | `actor_rollout_ref.rollout.checkpoint_engine.backend` |
| 新 trainer 模式 | `ppo/v1/trainer_base.py::register_trainer` + 覆写 `on_*` hook | `trainer.v1.trainer_mode` |
| 自定义 AgentLoopManager | 实现 `generate_sequences` 并把输出写入 TransferQueue | `rollout.agent.agent_loop_manager_class=<fqn>` |
| 新 dispatch 模式 | `decorator.py::register_dispatch_mode` | worker 方法上的 `@register(dispatch_mode=...)` |
| 新 Ray role / worker | `trainer/ppo/utils.py::Role` + `RayClassWithInitArgs` | 在 trainer `_init_resource_pool_mgr` 中映射 resource pool |

当前注册的优势估计器（`AdvantageEstimator`）：`gae, grpo, grpo_vectorized, gdpo, grpo_passk,
reinforce_plus_plus, reinforce_plus_plus_baseline, remax, rloo, rloo_vectorized, opo, gpg,
optimal_token_baseline, tir_optimal_token_baseline`。

当前注册的 policy loss：`vanilla, dppo_tv, dppo_kl, gspo, sapo, gpg, clip_cov, kl_cov, geo_mean,
dro, cispo, bypass_mode`。`loss_agg_mode`：`token-mean, token-sum, seq-mean-token-sum,
seq-mean-token-sum-norm, seq-mean-token-mean`。KL 估计（`kl_penalty` / `kl_loss_type`）：
`kl|k1, abs, mse|k2, low_var_kl|k3, full`，加 `+` 后缀（如 `k3+`）启用 straight-through k2 梯度。

## 配置（Hydra）

主配置 `verl/trainer/config/ppo_trainer.yaml` 由 defaults list 组装：

- `model_engine: dp`（默认 FSDP/FSDP2）；`model_engine=megatron|veomni|torchtitan` 一次切换
  actor/ref/critic 的 `${model_engine}_actor.yaml` 等子配置。`ppo_megatron_trainer.yaml` 已
  deprecated，等价于 `model_engine=megatron`。
- 子配置目录：`actor/ critic/ ref/ rollout/ model/ engine/ optim/ data/ reward/ algorithm/
  distillation/ transfer_queue/ profiler/`。
- `_generated_ppo_*_trainer.yaml` 是展开后的完整配置快照（由 `scripts/generate_trainer_config.sh`
  生成，CI 校验）——**查默认值时读这几份最快**。
- YAML 格式规则（CI 强制）：注释写在字段上方、字段间空行、禁止行尾注释。

关键字段与语义：

| 字段 | 含义 |
|---|---|
| `data.train_batch_size` | 每 step 的 **prompt** 数 |
| `actor_rollout_ref.rollout.n` | 每 prompt 采样数（GRPO 组大小） |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 以 prompt 计；trainer 内部会 `× rollout.n` |
| `ppo_micro_batch_size_per_gpu` / `use_dynamic_bsz` + `ppo_max_token_len_per_gpu` | 每卡前后向粒度；动态 bsz 按 token 预算打包 |
| `rollout.log_prob_*`, `ref.log_prob_*` | log-prob 重算阶段的 micro batch / token 预算 |
| `rollout.name / tensor_model_parallel_size / gpu_memory_utilization / free_cache_engine` | 推理后端、TP、显存占比、sleep 时释放 KV |
| `rollout.engine_kwargs.{vllm,sglang}.*` | 透传给推理引擎（如 `+...sglang.attention_backend=flashinfer`） |
| `actor.fsdp_config.{param_offload,optimizer_offload,ulysses_sequence_parallel_size}` | FSDP 卸载、Ulysses SP |
| `actor.megatron.{tensor,pipeline,context,expert}_model_parallel_size` | Megatron 并行度 |
| `model.use_remove_padding`, `model.enable_gradient_checkpointing` | 序列打包、重计算 |
| `actor.use_kl_loss` / `algorithm.use_kl_in_reward` | 二者任一为 True 才会启用 ref policy |
| `trainer.balance_batch` | 按序列长度在 DP rank 间重排（`seqlen_balancing.py`） |
| `trainer.use_v1`, `trainer.v1.trainer_mode` | V1/V0 选择与同步/异步模式 |

`verl/utils/config.py::validate_config` 中最常触发的断言：非动态 bsz 时
`train_batch_size × rollout.n` 必须能被 `n_gpus`（Megatron 为 `dp × micro_bsz`）整除；Megatron 下
`n_gpus` 必须能被 `TP × PP × CP` 整除；micro batch 的 `param` 与 `param_per_gpu` 至少设一个；
V1 `separate_async` 要求 `train_batch_size == parameter_sync_step × ppo_mini_batch_size`。

典型启动（截自 `examples/grpo_trainer/run_qwen3_8b_fsdp.sh`）：

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$HOME/data/gsm8k/train.parquet']" data.val_files="['$HOME/data/gsm8k/test.parquet']" \
  data.train_batch_size=1024 data.max_prompt_length=1024 data.max_response_length=2048 \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=5 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1
# Megatron: 追加 model_engine=megatron actor_rollout_ref.actor.megatron.tensor_model_parallel_size=...
```

上游脚本默认通过 `uv run --frozen --all-packages --extra <vllm|sglang> --extra fsdp` 启动，并把
同一命令写入 `ray_kwargs.ray_init.runtime_env.py_executable` 让 Ray worker 使用同一环境；
`VERL_USE_UV=0` 回退到系统 Python。环境管理见 `VERL_REPO/manage_envs.py` 与 `pyproject.toml` extras
（`vllm, sglang, fsdp, megatron, veomni, torchtitan, cpu, ...`）。

## 数据与 Reward

- 数据为 parquet，`RLHFDataset`（`utils/dataset/rl_dataset.py`）默认读取 `prompt`（chat messages，
  可用 `data.prompt_key` 改），并携带 `data_source`、`reward_model.ground_truth`、`extra_info`、可选
  `agent_name`。预处理参考 `examples/data_preprocess/`（gsm8k.py, math_dataset.py, geo3k.py,
  gsm8k_multiturn_w_tool.py, gsm8k_tool_agent_loop.py, preprocess_search_r1_dataset.py ...）。
- 默认规则打分 `utils/reward_score/__init__.py::default_compute_score` 按 `data_source` 分派：
  `openai/gsm8k`、MATH 系列、`math_dapo`/`aime*`、numina_*、代码（codecontests/apps/codeforces/taco，
  需 sandbox）、`hiyouga/geometry3k`、searchR1_*。未匹配会 `NotImplementedError`——这是新数据集
  最常见的报错，解决办法是设置 `reward.custom_reward_function`。
- 模型打分（GenRM / 判别式 RM）：`reward.reward_model.enable=True` + `reward.reward_model.rollout.*`；
  独立资源池用 `enable_resource_pool=True` + `n_gpus_per_node/nnodes`。
- 顶层 `custom_reward_function` 与 `reward_model.*`（`legacy_reward_impl.yaml`）是兼容旧配置的
  legacy 入口，新配置写在 `reward.*` 下。

## Search Strategy

**用 rg 精准检索，不要整文件加载**（`ray_trainer.py` 1.8k 行、`core_algos.py` 2.5k 行、
`trainer_base.py` 2k 行）。

```bash
VERL_REPO="$HOME/.cursor/skills/verl-skill/repos/verl"   # 替换为实际路径

# 优势估计器与 policy loss 注册表
rg -n "@register_adv_est|@register_policy_loss" $VERL_REPO/verl/trainer/ppo/core_algos.py

# V1 trainer 单步流水线与 hook
rg -n "def _step_once|def on_[a-z_]+\(" $VERL_REPO/verl/trainer/ppo/v1/

# Worker 上可被 driver 调用的方法及其 dispatch 模式
rg -n -A1 "@register\(" $VERL_REPO/verl/workers/engine_workers.py

# 训练后端注册
rg -n "@EngineRegistry.register" $VERL_REPO/verl/workers/engine/

# 推理后端 / 副本注册
rg -n "_ROLLOUT_REGISTRY|RolloutReplicaRegistry.register" $VERL_REPO/verl/workers/rollout/

# 权重同步（hybrid 与 checkpoint engine）
rg -n "async def update_weights|send_weights|sleep_replicas" $VERL_REPO/verl/workers/engine_workers.py $VERL_REPO/verl/checkpoint_engine/base.py

# Agent loop / 工具 / 工具解析器注册
rg -n "@register\(\"|ToolParser.register" $VERL_REPO/verl/experimental/agent_loop/

# Reward manager 注册（新旧两套）
rg -n "@register\(\"" $VERL_REPO/verl/experimental/reward_loop/reward_manager/ $VERL_REPO/verl/workers/reward_manager/

# 配置默认值（展开后快照）
rg -n "ppo_max_token_len_per_gpu|gpu_memory_utilization" $VERL_REPO/verl/trainer/config/_generated_ppo_trainer.yaml

# 配置校验报错定位
rg -n "assert|raise ValueError" $VERL_REPO/verl/utils/config.py

# 训推概率差指标
rg -n "rollout_probs_diff" $VERL_REPO/verl/utils/debug/metrics.py
```

## 常见开发场景

### 新增 RL 算法（优势估计 + loss）

1. 在 `core_algos.py` 用 `@register_adv_est("my_adv")` 实现
   `fn(token_level_rewards, response_mask, index, config, **kw) -> (advantages, returns)`，
   参考 `compute_grpo_outcome_advantage`。
2. 如需新目标函数，用 `@register_policy_loss("my_loss")`，参考 `vanilla` / `gspo` / `cispo` 的签名
   （`old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights`）。
3. 配置 `algorithm.adv_estimator=my_adv`、`actor_rollout_ref.actor.policy_loss.loss_mode=my_loss`；
   若需要 critic，确认 `trainer/ppo/utils.py::need_critic` 的判断。
4. 在 `tests/trainer/` 补 `*_on_cpu.py` 单测；算法文档放 `docs/algo/`。

### 多轮工具调用 / Agentic RL

1. 数据集加 `agent_name="tool_agent"`（或自定义名），工具配置 yaml 指向 `BaseTool` 子类
   （`class_name` + OpenAI function schema），见 `examples/data_preprocess/gsm8k_tool_agent_loop.py`。
2. 设置 `actor_rollout_ref.rollout.multi_turn.enable=True`、`tool_config_path`、`format`、
   `max_assistant_turns`。
3. 自定义交互逻辑：继承 `AgentLoopBase` 实现 `async run(sampling_params, **kw) -> AgentLoopOutput`
   （`response_mask` 中工具 token 置 0）；`@register("my_agent")`。
4. 文档：`docs/advance/agent_loop.rst`、`docs/start/agentic_rl.rst`、`docs/sglang_multiturn/`。
5. 模板一致性：`scripts/chat_template_checker.py` 校验 chat template 与 continuous token。

### 切换训练 / 推理后端

- FSDP → Megatron：`model_engine=megatron` + `actor/ref.megatron.*` 并行度；HF→mcore 权重可用
  `scripts/converter_hf_to_mcore.py`，或 `actor_rollout_ref.actor.megatron.use_mbridge=True`。参考 `examples/grpo_trainer/*_megatron.sh`。
- vLLM ↔ SGLang ↔ TRT-LLM：`actor_rollout_ref.rollout.name=...`；SGLang 侧内部问题转 `sglang-skill`。
- Blackwell（GB200/B200）脚本经验（`run_qwen3_8b_fsdp.sh` 注释）：`rollout.enforce_eager=True`、
  FSDP `model_dtype=bfloat16`、SGLang 用 `attention_backend=flashinfer`（SM>90 无 FA3）。

### 异步 / 分离式训练

- V1：`trainer.v1.trainer_mode=colocate_async`（共置 + partial rollout）或 `separate_async`
  （`actor_rollout_ref.hybrid_engine=False` 时完全不在训练卡上起 rollout）；`docs/advance/v1_async_trainer.md`。
- 分离式权重同步走 `checkpoint_engine`（nccl / nixl / mooncake ...），带宽与选型见
  `verl/checkpoint_engine/README.md`；增量同步见 `docs/advance/delta_weight_sync.md`。
- 其他方案：`experimental/fully_async_policy/`、`experimental/one_step_off_policy/`。

### LoRA、SFT、蒸馏、评测、导出

- LoRA RL：`docs/advance/ppo_lora.rst`、`examples/tuning/lora/`；LoRA 时 ref 由 actor 关闭 adapter 得到。
- SFT：`torchrun -m verl.trainer.sft_trainer`（配置 `sft_trainer_engine.yaml`）；`examples/sft/`。
- 在线蒸馏：`distillation.*` 配置 + `experimental/teacher_loop/`；`docs/algo/opd.md`、
  `docs/advance/async-on-policy-distill.md`。
- 导出 HF：`python -m verl.model_merger merge --backend fsdp|megatron|veomni --local_dir <ckpt>/actor --target_dir <out>`。

## 训推一致（Training-Inference Mismatch）

推理引擎（vLLM/SGLang）与训练引擎（FSDP/Megatron）kernel、精度、batch 组成不同，导致
rollout log-prob 与训练重算 log-prob 偏离，可能引发 RL collapse。verl 的相关能力：

| 能力 | 入口 |
|---|---|
| 观测偏差 | 指标 `training/rollout_probs_diff_{mean,max,std}`、`training/rollout_actor_probs_pearson_corr`（`utils/debug/metrics.py`）；依赖 `rollout.calculate_log_probs`（默认 True）记录的 `rollout_log_probs` |
| 修正（TIS / IcePop / 拒绝采样） | `algorithm.rollout_correction.{rollout_is, rollout_is_threshold, rollout_rs, rollout_rs_threshold, bypass_mode, loss_type}`；实现 `trainer/ppo/rollout_corr_helper.py`；文档 `docs/algo/rollout_corr.md`、`rollout_corr_math.md`；示例 `examples/rollout_correction/` |
| rollout log-prob 口径 | `actor_rollout_ref.rollout.logprobs_mode`（默认 `processed_logprobs`） |
| bitwise 可复现 | `rollout.full_determinism=true` + `actor/ref.<engine>_config.full_determinism=true` + `rollout.seed`；**必须 `trainer.use_v1=false`**（V1 按完成顺序收集 TQ 数据）；依赖 vLLM batch invariance，未覆盖时 `max_num_seqs=1`；文档 `docs/advance/determinism.md` |
| MoE 路由一致 | Router replay：`rollout.enable_rollout_routing_replay` + engine `router_replay`；`utils/megatron/router_replay_*.py`、`utils/veomni/router_replay.py`；示例 `examples/router_replay/` |
| score centering | `algorithm.rollout_correction.score_centering`（仅 V1 + FSDP + vLLM + `topk_log_probs>0`，见 `validate_config`） |

排查顺序：先看 `rollout_probs_diff` 是否随 step 增长 → 对齐 dtype / logprobs_mode / 采样参数
（temperature、top_p）→ MoE 开 router replay → 再考虑 rollout correction；需要 kernel 级对齐时
转 `gpu-performance-router`。

## 性能与显存排查

优先依据 `docs/perf/perf_tuning.rst`（rollout 调优、remove padding、batch size、dynamic bsz、
Ulysses SP、LigerKernel、FSDP forward prefetch、FSDP2 迁移、entropy 显存优化）与
`docs/perf/best_practices.rst`、`docs/perf/device_tuning.rst`（各规模模型所需卡数与配置）。

1. **先看分段计时**：每 step 日志 `timing_s/{gen,reward,old_log_prob,ref,values,adv,update_critic,update_actor,update_weights}`，
   判断瓶颈在 rollout 还是训练。
2. **Rollout 慢**：调 `gpu_memory_utilization`、`tensor_model_parallel_size`、`max_num_batched_tokens`、
   `enable_chunked_prefill/prefix_caching`、关闭 `enforce_eager` 以用 CUDA Graph；长尾 rollout
   考虑 V1 async 模式。
3. **训练慢 / OOM**：`use_remove_padding=True`、`use_dynamic_bsz=True` + 合理 `*_max_token_len_per_gpu`、
   `enable_gradient_checkpointing`、`entropy_from_logits_with_chunking`、`use_fused_kernels`、
   param/optimizer offload、Ulysses SP 或 Megatron CP。
4. **权重同步慢**：`update_weights` 段；调 `rollout.checkpoint_engine.update_weights_bucket_megabytes`，
   分离式选择更合适的 checkpoint engine 后端。
5. **Profiling**：`global_profiler.tool=nsys|torch|torch_memory|npu`、`global_profiler.steps=[...]`，
   并在角色上开启 `actor_rollout_ref.actor.profiler.enable=True`（`all_ranks`/`ranks`）。nsys 选项在
   `global_profiler.global_tool_config.nsys.{controller,worker}_nsight_options`。文档
   `docs/perf/nsight_profiling.md`、`torch_profiling.md`、`verl_profiler_system.md`；示例 `examples/profile/`。
   拿到 `.nsys-rep` 后转 `nsys-profile-analysis`；具体 kernel 转 `gpu-performance-router`。

## 调试

- Ray 分布式断点：`docs/start/ray_debug_tutorial.rst`（Ray Distributed Debugger VSCode 插件）。
- 环境自检：`python scripts/diagnose.py`（OS/硬件/pip/verl/网络）。
- 打印展开后配置：`scripts/print_cfg.py`；查看 rollout 落盘：`trainer.rollout_data_dir` +
  `scripts/rollout_viewer.py`。
- 单测：`tests/` 子目录与 `verl/` 子包一一对应；`*_on_cpu.py` 跑 CPU，其余默认需要 GPU；
  `tests/special_{distributed,e2e,npu,sanity,standalone}` 为特殊环境。运行：`pytest tests/trainer/<file>_on_cpu.py -q`。
- FAQ：`docs/faq/faq.rst`；多节点：`docs/start/multinode.rst`；AMD：`docs/amd_tutorial/`；
  昇腾：`docs/ascend_tutorial/`。

## 向上游贡献

上游 `VERL_REPO/AGENTS.md` 对 AI 辅助贡献有强制规则：提 PR 前用 `gh` 检查重复 issue/PR；禁止
零碎 busywork PR；**纯 agent PR 不允许**，提交人须逐行审阅并运行相关测试；PR 描述必须写明非重复
原因、测试命令与结果、声明使用了 AI；commit 加 `Co-authored-by:` trailer；违反可能被封禁。
环境使用 `uv venv --python 3.12` + `pre-commit install`。

## Additional References

- 文档索引：`VERL_REPO/docs/index.rst`；核心概念 `docs/hybrid_flow.rst`、`docs/single_controller.rst`、
  `docs/workers/{engine_workers,model_engine,ray_trainer}.rst`、`docs/examples/ppo_code_architecture.rst`、
  `docs/examples/config.rst`；数据面 `docs/data/transfer_queue.md`；放置策略 `docs/advance/placement.rst`。
- 在线文档: https://verl.readthedocs.io/
- GitHub: https://github.com/verl-project/verl
- 本地知识库中的 verl 工程经验文章：`zhihu-knowledge-base`（搜索 "verl"）。
