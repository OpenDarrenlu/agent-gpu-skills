# Linear Attention 系列 State Layout：从 SMEM Bank Conflict 看 k-last vs v-last

**作者**: NobodyLove the life you live.

**原文链接**: https://zhuanlan.zhihu.com/p/2031471761737430946

---

摘要

Aside: 之前组里 intern 问了我什么原因选择 k-last，隔了两月我自己都差点忘了，记录一下供自己备忘。

TLDR

在 LA decode 路径中，state 采用 k-last (BHVK) 相比 v-last (BHKV) 的根本优势来自 O 计算时 SMEM bank conflict 的天然避免。虽然 state 本身的更新在数学上对 v/k 近似对称，但输出计算 o[v] = ∑_k h[v,k] · q[k] 需要沿 K 维规约，使得 k-last 布局显著更友好。

1. State 更新的对称性

状态更新：

h[v, k] \leftarrow g \cdot h[v, k] + k_t[k] \otimes v_t[v]

从这个公式看，v 和 k 是对称的——布局为 BHVK (k-last) 或 BHKV (v-last) 都能接受更新。

2. 输出计算的不对称性与 SMEM Bank Conflict

关键差异在输出计算：

o[v] = \sum_k h[v, k] \cdot q_t[k]

这一步要在 K 维做规约。State 体积较大，驻留在 SMEM 而非寄存器，会被反复读取。哪种布局让”固定 v、沿 k 扫描”的 SMEM 访问更连续，性能就更好。

2.1 Bank Conflict 对比

CUDA SMEM 有 32 个 bank。访问地址的 bank 映射为：

bank = (addr) \bmod 32

k-last (BHVK) 的 k 连续：

每个 warp 的 32 个 lane 扫描 32 个连续 k 值

lane_i 访问 addr = v · TILE_K + i

bank_i = (v · TILE_K + i) mod 32 ≈ 均匀分布

结果：32 条访存几乎无冲突

v-last (BHKV) 的 v 不连续（K 连续）：

固定 v 扫 k 时，lane_i 访问 addr = k · V + v

bank_i = (lane · V + v) mod 32

当 V=128 时，bank_i = v mod 32（所有 lane 落到同一 bank）

结果：最坏冲突模式
3. v-last 通过额外机制也能可行

需要注意的是，v-last 布局完全可以支持，但需要引入额外的缓解措施：

3.1 Padding 与 Swizzle

最常见的方法是对 SMEM 进行 padding 或 swizzle：

Padding：在 state 的某一维后加上冗余元素，改变 bank 映射周期，破坏最坏冲突
Swizzle：通过地址变换（如 XOR 高位与低位）重新打乱 bank 分配

例如将 SMEM 从 [TILE_K, TILE_V] 改为 [TILE_K, TILE_V + PAD]，可让 32 个 lane 的 bank 分配从”全集中”变为”分散”。

3.2 权衡
k-last：天然高效，无需额外机制
v-last：可行但需额外成本，最终可能不如 k-last 直接

因此在当前实现下，decode k-last 是零成本的最优选择，而 Prefill 输出 k-last / v-last 代价变化不大，考虑到长文本的今天，还是优先选择 k-last layout 较好。
