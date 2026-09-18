# FlashAttention-3 for Inference: INT8 Quantization and Query Head Packing for MQA/GQA (External)

**URL:** https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/
**Date:** November 27, 2024
**ISO Date:** 2024-11-27T17:17:23-08:00
**Categories:** 
**Tags:** 
**Content Type:** blog
**PDF URL:** 
**PDF Filename:** 
**Scraped At:** 2026-06-04T04:33:33.939752

---

## Excerpt

In thisblog postpresented on the Character.AI research blog, we explain two techniques that are important for usingFlashAttention-3for inference:

---

## Content

In this
blog post
presented on the Character.AI research blog, we explain two techniques that are important for using
FlashAttention-3
for inference:
A general methodology for in-kernel pre-processing of tensors via warp specialization, applied to the case of a half INT8 attention kernel design that upcasts the V tensor in the producer warpgroup.
Query head packing of the Q tile done for multi-query attention (MQA) or grouped query attention (GQA), which is needed to saturate bandwidth during the memory-bound decoding phase of inference.
We also give microbenchmark results for both prefill and decode-type attention workloads, measured on an NVIDIA H100 SXM5 GPU.
Optimizing AI Inference at Character.AI (Part Deux)
At Character.AI, we’re building personalized AI entertainment. In order to offer our users engaging, interactive experiences, it’s critical we achieve highly efficient inference, or the process by which LLMs generate replies. Our last post on this topic looked at several techniques that contribute to the performance and sustainability
Joint work with Character.AI
.
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
