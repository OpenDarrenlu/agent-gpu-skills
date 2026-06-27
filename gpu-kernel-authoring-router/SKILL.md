---
name: gpu-kernel-authoring-router
description: >
  Router for writing or modifying NVIDIA GPU kernels. Use for CUDA C++, PTX,
  Triton, Gluon, CUTLASS, CuTe, CuTeDSL, GEMM, attention kernels, reductions,
  fused ops, Tensor Core, WGMMA, TMA, persistent kernels, FP8/FP4, Blackwell/Hopper
  kernels, or choosing CUDA vs Triton vs CUTLASS. 中文触发词：写 GPU kernel、写 CUDA、
  写 Triton、写 CUTLASS、写 GEMM、写 attention kernel、算子实现、CUDA/Triton/CUTLASS 选型。
---

# GPU Kernel Authoring Router

Use this router when the user wants implementation guidance or code for GPU kernels.

## Route

- **CUDA C++ / PTX / low-level architecture/API questions**: use `cuda-skill`.
- **Python DSL kernels, Triton, Gluon, tl.* APIs, autotune**: use `triton-skill`.
- **CUTLASS/CuTe/CuTeDSL templates, GEMM builders, layout algebra**: use `cutlass-skill`.
- **Persistent kernel design or scheduling**: use `persistent-kernel-scheduling`.
- **Need theoretical performance before coding**: use `nv-gpu-kernel-performance-modeling`.
- **Need reference articles or reading order**: use `colfax-research-skill`.

## Workflow

1. Determine the implementation target: CUDA C++, Triton/Gluon, or CUTLASS/CuTe.
2. If the target is unclear, choose the smallest practical stack:
   - Triton for fast Python-level custom kernels.
   - CUTLASS/CuTe for production GEMM/attention-like templates.
   - CUDA/PTX for architecture-specific control or unsupported patterns.
3. Read the chosen downstream skill before giving API names, code structure, or file paths.
4. Include a verification path: correctness test first, then profiling plan if performance matters.
