# 从 SASS 恢复语义与布局

## 调用契约

从 caller、Driver/Runtime launch、框架参数和 profile 建立 host→device 关联。参数表包含序号、device-side offset、size、已知 alignment、传递方式、标量位宽/符号、pointer 类型、shape/stride、读写、alias 和证据。按值结构体可能包含 pointer；TMA/tensor map、texture/surface handle 不是普通数组地址。

支持时用 `cuFuncGetParamInfo` 核对 offset/size，但它不提供含义。读 `.nv.info` 的 KPARAM 或 constant bank 时，**不能跨 SM/ABI 硬编码参数起点（如 0x160）**；bank 还可能含 launch/runtime 数据。

恢复 grid/block、动态/静态 smem、cluster、cooperative launch、函数属性、常量/global 初始化、workspace/counter/RNG、依赖函数/重定位。区分数学输入域、本二进制的合法特化域和当前测试覆盖域；不要在原 kernel 不支持的尾块上宣称验证失败。

## 反向切片与精确模型

从 STG、global atomic/reduction、surface store、全局状态和退出条件回溯值/地址。建立有 PC 的 CFG、显式谓词和近似 SSA；物理寄存器会在不同 live range 承载不同含义。

- 标记 `R7_3:u32`、`addr:u64=concat(hi,lo)`，保留 carry、guard、predicate 与 reconvergence。
- 区分 R/UR、P/UP、RZ/PT、特殊寄存器，以及 local/shared/global/constant。
- IMAD/LEA/IADD3 按完整 modifier、符号、宽度、进位解释；wraparound/high-half/符号扩展不是无界整数运算。
- LOP3 按 LUT 真值表，PRMT/pack/bitfield 按 selector；相似名称或 pseudo-op 不代表相同语义。
- 建立 branch/loop/call/return/间接目标的路径条件；reconvergence 不自动等于内存 barrier。
- 未解指令用 opaque 节点，记录未知副作用；输出依赖 opaque 时不能宣布完整语义已恢复。

保持原始精确模型与高层简化模型的对应关系。Ghidra C 风格输出和 lifter IR 是候选模型，不是可直接提交的高性能 CUDA。

## 地址与 lane 布局

恢复每次访问的 byte address，例如：

```text
lane = linear_thread_id % 32; warp = linear_thread_id / 32
address = base + element_bytes * logical_index + byte_subfield
logical_index = f(ctaid.xyz, tid.xyz, loop_counter, strides, scalar_params)
```

这是表达形式，不是假定 block 一维。linear id 按 CUDA 多维规则展开，区分 byte/element offset，核对 vector width/mask/alignment。

选跨 8/16/32-lane 边界的 lane、多个 warp 和相邻循环，展开地址表，核对覆盖/重复/coalescing/转置/bank 分布/swizzle。XOR 要连同 smem base、stage offset 和 element size 推导，不能凭 LOP3 猜 CuTe Layout。

先用输出 store 确定 accumulator 的 lane→element 映射，再追溯 MMA 和输入 load。shared 的生产者/消费者视图可能不同，证明两边地址匹配，不强求索引表达相同。

## 集合操作与数值

归约/scan/shuffle/vote 恢复 active mask、参与 lanes、归约树、分段边界、结果 lane。CTA barrier 不能仅由部分 CTA 进入；warp/CTA/cluster 同步不能互换。

记录输入/累加/输出 dtype、舍入、FMA contraction、FTZ、approx math、TF32、saturation、NaN/Inf/±0、量化 scale/zero point/pack、stochastic RNG。fast math 和归约重排会改变数值。

| 模式线索 | 候选 | 必须补证 |
|---|---|---|
| affine index + FFMA + store | pointwise | 参数、边界、alias |
| SHFL/REDUX + 树状组合 | reduction/scan | mask/operator/结果映射 |
| EX2 + max/sum + reciprocal | softmax/指数运算 | 同一依赖链、归约域、归一化、mask |
| 两组 MMA + 中间指数归约 | attention fusion | Q/K/V、scale、mask、在线 max/sum 更新 |
| 领取 counter/tile + loop | persistent scheduler | 初始化、粒度、终止、跨 CTA 可见性 |

## 受控实验

在已确认合法域内使用 one-hot、各维不同 ramp、小整数、符号变化、sentinel、随机/重复输入。组合不同维度与分布区分转置、stride 和标量类型，不能只用一个样本。

编译最小 CUDA/PTX snippet，每次只改变 dtype/layout/vector width/unroll/rounding/target，保存源码、flags、PTX、cubin、SASS。使用匹配的 compiler/SM；关闭优化只辅助理解，不能替代原始性能结构。

局部模拟只验证覆盖的语义，跨 warp/CTA 协议仍需运行验证。每条假设记录可推翻它的观测；遇到矛盾先改模型，不靠扩大容差消除错误。
