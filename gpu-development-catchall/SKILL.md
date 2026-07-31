---
name: gpu-development-catchall
description: >
  |
  Catch-all entry point for any GPU, NVIDIA CUDA, AMD ROCm/CDNA, AI infrastructure, HPC, or
  performance-related question. Use this skill when the user asks anything about
  GPU programming, CUDA kernels, PTX, Triton, CUTLASS, CuTe, DeepGEMM, MoE,
  LLM serving, SGLang, vLLM, KV cache, attention, NCU/NSYS profiling, GPU
  communication, RDMA, NCCL, NVSHMEM, multi-GPU training/inference, B200/H100/A100,
  Blackwell/Hopper/Ampere, AMD Instinct, CDNA4, ROCm, HIP, gfx950, Tensor Core, SM occupancy, warp stalls, memory bandwidth,
  CUDA Graph, persistent kernels, Stream-K, CLC, TMA, WGMMA, tcgen05, FP8/FP4/BF16,
  quantization, or anything that sounds like "GPU 相关" / "性能优化" / "kernel 怎么写" /
  "为什么慢" / "多卡通信" / "推理服务" / "算子实现". This skill does not answer
  directly; it routes to the most specific skill in the agent-gpu-skills bundle.
  中文触发词：GPU、CUDA、NVIDIA、算子、kernel、性能、profiling、ncu、nsys、多卡、
  通信、推理、LLM、MoE、GEMM、attention、Triton、CUTLASS、DeepGEMM、SGLang、慢、优化、
  并行、HBM、L2、shared memory、寄存器、occupancy、wave、latency、throughput。
triggers:
  - "AMD"
  - "CDNA4"
  - "ROCm"
  - "HIP"
  - "gfx950"
  - "CUDA"
  - "CUTLASS"
  - "DeepGEMM"
  - "GEMM"
  - "GPU"
  - "LLM"
  - "MoE"
  - "NVIDIA"
  - "SGLang"
  - "Triton"
  - "attention"
  - "kernel"
  - "ncu"
  - "nsys"
  - "profiling"
  - "为什么慢"
  - "优化"
  - "分析下"
  - "多卡"
  - "如何实现"
  - "帮忙看一下"
  - "怎么写"
  - "怎么这么慢"
  - "性能"
  - "推理"
  - "推荐"
  - "有没有资料"
  - "算子"
  - "解释下"
  - "讲一下"
  - "通信"
---

# GPU Development Catch-All Router

This skill exists to maximize the hit rate of the `agent-gpu-skills` bundle.
When the user's query touches anything GPU-related, use this skill first to pick
a more specific downstream skill.

## Routing Table

| If the user asks about... | Use this skill |
|---|---|
| AMD/ROCm/HIP docs, libraries, tools, broad local AMD knowledge | `amd-gpu-docs` |
| AMD Instinct, CDNA4 ISA, gfx950, AMD GPU assembly | `amd-instinct-cdna4-isa` |
| Writing/modifying CUDA, PTX, Triton, CUTLASS, CuTe, DeepGEMM kernels | `gpu-kernel-authoring-router` |
| A kernel is slow, performance bottleneck, NCU/NSYS, profiling | `gpu-performance-router` |
| SGLang, LLM serving, KV cache, attention backend, throughput/latency | `llm-serving-router` |
| NCCL, NVSHMEM, multi-GPU/multi-node communication, RDMA, collectives | `gpu-communication-libraries` |
| CUDA API, PTX ISA, architecture, occupancy, memory, CUDA tools | `cuda-skill` |
| Triton/Gluon Python kernels | `triton-skill` |
| CUTLASS/CuTe/CuTeDSL templates and GEMM | `cutlass-skill` |
| DeepGEMM grouped/MoE GEMM | `deepgemm-skill` |
| SGLang runtime/serving | `sglang-skill` |
| Analytical performance modeling, roofline, latency prediction | `nv-gpu-kernel-performance-modeling` |
| Persistent kernel scheduling, CLC, Stream-K, tail effect | `persistent-kernel-scheduling` |
| Persistent kernel utilization, SM/TC bubbles, setmaxnreg | `persistent-kernel-utilization` |
| NCU diagnosis of persistent kernels | `ncu-persistent-kernel-diagnosis` |
| NCU report analysis, .ncu-rep, source/SASS | `ncu-report-skill` |
| Colfax Research articles and reading order | `colfax-research-skill` |

## Workflow

1. Read the user's query and identify the broad domain (kernel writing, performance,
   serving, communication, or reference).
2. Route to the matching router or direct skill above.
3. Do not answer the question from this skill alone; always delegate to a downstream
   skill that has the actual references and source code maps.
