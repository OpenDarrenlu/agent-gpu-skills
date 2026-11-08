---
name: gpu-framework-router
description: Route GPU framework, runtime, serving, RL post-training, and framework-level profiling questions to the right skill. Use for SGLang, vLLM-like serving, verl/HybridFlow RL training (PPO/GRPO, rollout, reward, agent loop), PyTorch runtime behavior, attention backends, KV cache, end-to-end latency, workload benchmarks, Nsight Systems traces, framework-level Nsight Compute profiling, or deciding whether a problem belongs to the framework or the underlying kernel.
---

# GPU Framework Router

Use this entry point when the question is about a GPU framework or an end-to-end
runtime rather than a standalone kernel.

## Routing order

1. **SGLang internals, model/backend integration, KV cache, scheduler, or serving
   regressions**: use `sglang-skill`.
2. **RL post-training with verl/HybridFlow** (PPO/GRPO/DAPO trainers, Ray single
   controller, FSDP/Megatron model engines, vLLM/SGLang rollout integration,
   actor-to-rollout weight sync, reward functions, agent loops, async RL,
   train-inference mismatch, RL step OOM/throughput): use `verl-skill`. If the
   problem is inside the SGLang rollout server itself, continue with
   `sglang-skill`; weight-sync collective hangs go to `nccl-skill`.
   For the end-to-end engineering workflow that makes training and vLLM rollout
   logprobs bitwise identical (true on-policy RL: C0–C10 gates, batch-invariant
   kernels, training forward reusing inference kernels, backward/STE and
   quantization contracts, delivery and paired acceptance), use
   `zero-mismatch-rl` — the vendored Inferact/zero-mismatch-rl skill: full
   workflow, per-model recipes and failure records (DeepSeek-V4-Flash,
   DeepSeek-V4.1-Flash, Nemotron, Kimi-K3.1), weight update / online sync,
   checkpoint/resume and delivery. For the same workflow as vLLM's own
   `.agents/skills` snapshot with the pinned vLLM source map, use
   `train-inference-consistency`. Keep verl-specific metrics and config in
   `verl-skill`.
3. **SGLang serving-side failures, profiling or contributions** (KL
   prefill/decode consistency, CUDA crash, distributed hang, production
   incident triage, `--mamba-full-memory-ratio`, profile capture, adding
   JIT/AOT kernels, writing SGLang tests): use the vendored
   sglang-repo-skills skills — `kl-consistency-test`, `debug-cuda-crash`,
   `debug-distributed-hang`, `sglang-prod-incident-triage`,
   `compute-mamba-ratio`, `generate-profile`, `add-jit-kernel`,
   `add-sgl-kernel`, `kernel-organization`, `write-sglang-test`.
4. **LLM serving design, vLLM/SGLang/LMDeploy/TensorRT-LLM comparison, batching,
   capacity, TTFT/TBT, or deployment strategy**: use `llm-serving-router`.
5. **Framework-level workload profiling or benchmark design**: use
   `llm-serving-auto-benchmark` or `perf-workload-profiling` when installed.
   For a PyTorch baseline that will become a TileLang kernel, use
   `torch-profiling-tilelang-programs`.
6. **Whole-application CPU/GPU timeline, launch causality, overlap, or NVTX
   analysis**: use `perf-nsight-systems` or `nsys-profile-analysis`.
7. **A framework dispatches a slow individual CUDA/Triton/CUTLASS kernel**:
   hand off to `gpu-performance-router`, then use the most specific kernel skill.
8. **PyTorch profiler, compiler, or host-side runtime behavior**: use
   `llm-torch-profiler-analysis`, `torch-profiling-tilelang-programs`,
   `perf-host-analysis`, or `perf-host-optimization` when installed.

## Framework versus kernel

Ask one separating question before optimizing:

- Is end-to-end latency dominated by scheduling, batching, memory management,
  synchronization, launch overhead, or a backend choice? Stay in this router.
- Is one kernel or one dispatch path dominant after measurement? Hand off to
  `gpu-performance-router`.

Do not infer a framework bottleneck from a single kernel metric. Start with a
representative workload and a framework-level timeline, then drill down into
kernel profiling only after the hot path is established.
