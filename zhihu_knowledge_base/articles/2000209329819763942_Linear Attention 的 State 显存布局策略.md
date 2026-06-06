# Linear Attention 的 State 显存布局策略

**作者**: NobodyLove the life you live.

**原文链接**: https://zhuanlan.zhihu.com/p/2000209329819763942

---

尝试用 cutedsl 写一下 decode kernel，从 smem 友好性来看，最好 state 是以 B, T, V, K 的 row major 形式存储。

看了一眼 flashinfer 源码， gdn decode 给了两个版本，pretranspose & nontranspose 版本，linear attention 从之前写 prefill 的感觉，也是 V,K state 存储是友好的。

其实可以打通，一开始 prefill 就以 B,T,V,K 的形式存储 state，decode 这边也自然友好
