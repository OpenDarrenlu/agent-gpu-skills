# A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library

**URL:** https://research.colfax-intl.com/nvidia-hopper-flashattention-2/
**Date:** December 5, 2023
**ISO Date:** 2023-12-05T00:09:34-08:00
**Categories:** 
**Tags:** 
**Content Type:** pdf
**PDF URL:** https://research.colfax-intl.com/download/colfax-flashattention/?tmstv=1780547652
**PDF Filename:** colfax-flashattention.pdf
**Scraped At:** 2026-06-04T04:34:12.898660

---

## Excerpt

We provide an optimized implementation of the forward pass of FlashAttention-2, a popular memory-aware scaled dot-product attention algorithm, as a custom fused CUDA®kernel targeting NVIDIA Hopper™architecture and written using the open-source CUTLASS library. In doing so, we explain the challenges and techniques involved in fusing online-softmax with back-to-back GEMM kernels, utilizing the Hopper-specific Tensor Memory Accelerator (TMA) and Warpgroup Matrix-Multiply-Accumulate (WGMMA) instruct

---

## Content

We provide an optimized implementation of the forward pass of FlashAttention-2, a popular memory-aware scaled dot-product attention algorithm, as a custom fused CUDA
®
kernel targeting NVIDIA Hopper
™
architecture and written using the open-source CUTLASS library. In doing so, we explain the challenges and techniques involved in fusing online-softmax with back-to-back GEMM kernels, utilizing the Hopper-specific Tensor Memory Accelerator (TMA) and Warpgroup Matrix-Multiply-Accumulate (WGMMA) instructions, defining and transforming CUTLASS Layouts and Tensors, overlapping copy and GEMM operations, and choosing optimal tile sizes for the Q, K and V attention matrices while balancing the register pressure and shared memory utilization. In head-to-head benchmarks on a single NVIDIA
®
H100 Tensor Core PCIe GPU for some common choices of hyperparameters, we observe 20-50% higher FLOPs/s over a version of FlashAttention-2 optimized for last-generation NVIDIA Ampere architecture.
colfax-flashattention.pdf
Share this:
Share on LinkedIn (Opens in new window)
LinkedIn
Share on X (Opens in new window)
X
Share on Facebook (Opens in new window)
Facebook
Share on Reddit (Opens in new window)
Reddit
Like this:
Like
Loading…
