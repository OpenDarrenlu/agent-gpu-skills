# Tensor Core、tile 与异步流水线

## 按实际架构识别路径

以下是常见线索，不是完整 ISA 映射。读取目标 SM 和完整 modifier，再用相同 target 的 PTX/CUDA 实验确认。架构编号不是可用特性的简单超集，`a`/`f` 目标也有自己的兼容条件。

| 常见目标 | 可能的结构线索 | 重建重点 |
|---|---|---|
| SM80/86（Ampere） | HMMA/IMMA、LDSM、LDGSTS、DEPBAR/LDGDEPBAR | warp MMA fragment、cp.async 分组、smem stage、尾块 |
| SM89（Ada） | warp MMA 与扩展数据类型 | 精度、fragment、实际 toolkit lowering |
| SM90/90a（Hopper） | HGMMA、TMA 类 load、mbarrier 类操作、寄存器重分配 | WGMMA、warpgroup 协议、producer/consumer、TMA descriptor |
| SM100 系列（数据中心 Blackwell） | tcgen05 路径、TMEM、TMA、cluster/2-CTA 结构 | TMEM 生命周期、MMA 完成、CTA group、缩放格式 |
| SM120 系列（消费级 Blackwell） | HMMA/QMMA/OMMA 等不同 lowering | 从实际 warp MMA 路径推导；不要套 SM100 的 TMEM/tcgen05 |

从官方 PTX target notes 确认具体 opcode/格式是否支持。`FP8/FP4` 标签不足以说明累加精度、scale 格式、sparse metadata 或可用指令。

## 恢复 tile，而非只数 MMA

把 `M_tile/N_tile/K_step` 作为待推导量，用三个独立约束相互核对：

1. CTA/warp/warpgroup 到输出坐标的映射；每个参与 lane 的 accumulator 数、输出 store 布局。
2. 一个语义循环步中加载 A/B 的地址范围、元素数、K 指针增量、stage 空间。
3. 完整 MMA 形状、参与者、重复次数、accumulator 依赖与 unroll 因子。

同一 PTX MMA 可能展开成多条 SASS；一个静态循环也可能含多个 K step。计数需标注“每个什么单位”，排除 prologue/epilogue、predicate 不执行路径和重复统计。不能简单按 SASS count × 某一固定 FLOP 系数推导整个算法。

用带唯一坐标编码的合法输入、one-hot 和小整数定位 A/B orientation、fragment→C 映射；再用随机数据验证。遇到 sparse/block-scaled 路径，同时恢复 metadata/scale 地址、粒度、packing 和有效域。

## 重建异步依赖图

以每个 stage 的 buffer/phase 为单位画事件图：

```text
producer 获取空 stage → 发起 copy → 声明/完成到达
consumer 等待对应 phase → 读取 stage → 发起 MMA
等待 MMA 完成 → 标记 stage 可复用 → 下一轮
```

具体次序由原 binary 的证据决定，这只是核对框架。表中记录谁执行、操作范围、barrier id、期望 participant/transaction bytes、phase 翻转、buffer offset、初始状态。重点检查：

- cp.async commit/wait-group 和 CTA 数据可见性的关系；一个 warp 的 wait 不自动代替所有线程需要的同步。
- WGMMA fence/commit/wait 与 register/shared 操作的排序；异步 MMA 完成前不能覆写 operand 或读取未完成 accumulator。
- mbarrier 的初始化、arrival count、transaction byte count、phase/parity、尾轮处理。
- TMA descriptor 中 base、box shape、strides、swizzle、interleave、OOB 行为；caller 创建的 opaque tensor map 无法仅凭 UTMALDG 恢复。
- 普通 store 与 async proxy 之间所需的可见性；cluster 中 remote/shared 地址与参与 CTA。
- SM100 TMEM 的分配、列/地址布局、load/store、MMA 完成、释放；不能替换成普通 shared pointer。
- warp specialization 的角色分支、寄存器重分配、producer/consumer 到达关系；单看到 setmaxnreg 不能证明完整分工。
- persistent scheduler 的 tile 领取、split-K/Stream-K/workspace、退出和 global ordering；独立 launch 必须恢复初始状态。

## 映射到 CUDA / CUTLASS / CuTe

| 恢复出的结构 | 可选实现方向 | 核对点 |
|---|---|---|
| 一般 pointwise/reduction/不规则 gather | CUDA C++ + warp primitives | 位宽、mask、alias、归约顺序 |
| warp MMA、明确 shared/register tile | CuTe MMA/Copy atoms + Layout 或 CUDA inline PTX | 原子 shape、lane layout、alignment、smem swizzle |
| 规则 Hopper GEMM | CUTLASS 3.x collective mainloop/epilogue、明确 schedule | TileShape、ClusterShape、StageCount、kernel schedule、epilogue |
| 已证实 Blackwell tcgen05/TMEM | 当前 CUTLASS 对应 SM100 示例/collective | toolchain、TMEM、CTA group、scale 语义 |
| 融合 attention/自定义流水线 | CuTe + CUDA/局部 PTX，或明确适配的现有实现 | 在线 softmax、mask、rescale、双 MMA、同步与 epilogue |

从固定 CUTLASS commit 上与目标架构/数据类型一致的最小 example 出发，实际编译确认 API。不要凭记忆拼一个跨版本 `CollectiveBuilder`。用 auto schedule 可以快速建立基线，但性能重建时应记录最终选择；它不保证与原 kernel 相同。

候选参数结构与原 ABI 可不同，由各自 adapter 传递同一逻辑输入。需要的 packing、workspace 初始化和 descriptor setup 记入端到端范围；不能从候选总成本中偷偷删掉。

若关键结构只有 inline PTX 能控制，限制在小封装并保留编译/差分证据。CUDA/PTX 编译器不保证暴露每个 SASS scheduling bit；性能仍有差距时报告差距和可检验原因，不把 SASS patch 冒充 CUTLASS 源码。
