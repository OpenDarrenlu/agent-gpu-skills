# FlexAttention + FlashAttention-4: Fast and Flexible (External)

**URL:** https://research.colfax-intl.com/flexattention-flashattention-4-fast-and-flexible-external/
**Date:** March 10, 2026
**ISO Date:** 2026-03-10T19:58:16-07:00
**Categories:** 
**Tags:** 
**Content Type:** blog
**PDF URL:** 
**PDF Filename:** 
**Scraped At:** 2026-06-04T04:33:04.572317

---

## Excerpt

In this PyTorch blog on which we collaborated, we explain the FlexAttention extension to FlashAttention-4 (or from another point of view, the incorporation of FA-4 as an attention backend for the PyTorch FlexAttention API).

---

## Content

In this PyTorch blog on which we collaborated, we explain the FlexAttention extension to FlashAttention-4 (or from another point of view, the incorporation of FA-4 as an attention backend for the PyTorch FlexAttention API).
FlexAttention + FlashAttention-4: Fast and Flexible – PyTorch
On Hopper and Blackwell GPUs, FlexAttention now has a FlashAttention-4 backend.
We added support in PyTorch to automatically generate CuTeDSL score/mask modification functions, and to JIT-instantiate FlashAttention-4 for custom attention variants.
This leads to performance gains of 1.2× to 3.2× over the existing Triton implementation on compute-bound workloads.
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
