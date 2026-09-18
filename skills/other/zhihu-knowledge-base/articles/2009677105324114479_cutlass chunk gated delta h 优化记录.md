# cutlass chunk gated delta h 优化记录

**作者**: NobodyLove the life you live.

**原文链接**: https://zhuanlan.zhihu.com/p/2009677105324114479

---

updated-2:

persistent kernel + dynamic work stealing (atomic and double buffering index):

varlen perf geomean: 1.34x -> 1.50x




updated-1 26-02-26:

w,u,k,gk 改为3 stages，SMEM 大约 202KB < 228KB

结果：

指标	in_stage=2 (上次)	in_stage=3 (现在)	变化
bench_h_kernel geomean	1.60x	1.71x	+6%
bench_varlen geomean	1.25x	1.34x	+7%
测试	46/46 PASS	46/46 PASS	-

heavy workload 提升的整体会更大些




一、数据搬运优化 (TMA / GMEM → SMEM)
1. TMA 批量加载替代逐元素 GMEM 访问
U load: 将逐元素 GMEM 读取替换为 TMA G2S 批量加载 (edfc8f0)
v_new store: R2S → SMEM + TMA S2G 批量写回，替代逐元素 GMEM 写
gk load: TMA bulk load 替代 per-element GMEM 访问 (82d2462)
效果: geo mean 从 0.37x → 0.78x vs FLA（2.1x 提升）
2. Double-buffer TMA 流水线 (stages 1→2)
W、K^T 做 2-stage TMA load，Load warp 在 MMA 期间预取下一 chunk (bf3b9d2)
U 做 2-stage，fetch 与 CUDA 计算重叠
h_out、vnew_store 做 2-stage epilog 流水
效果: no-gating 1.06x → 1.38x（+23-27%），SMEM ~92KB → ~160KB（SM100 228KB 上限内）
3. gk SMEM 双缓冲
sGK ping-pong 预取：chunk N 的 Phase 3 预取 chunk N+1 的 gk (b126fd6)
Phase 1 不再阻塞等待 gk GMEM load
效果: gk-only 0.98x → 1.01x（首次超越 FLA）
二、计算重叠 (Overlap / Latency Hiding)
4. gk decay 延迟到 MMA overlap 窗口
初始将 gk decay (wait + exp2 + barrier + apply) 从 Phase 3 移到 Phase 1（WH MMA overlap window，K=128 > K=64，窗口更长）(d9aafce)
后续进一步延迟到 KV MMA overlap 窗口 (bfd5914)
效果: 累计提升 varlen 1.25x → 1.28x，non-varlen 1.56x → 1.59x
5. v_new R2T 先于 R2S（KV MMA 更早启动）
R2T v_new→TMEM 放在 R2S v_new→sVnew_store 之前，KV MMA 立即启动 (2646351)
效果: no-gating 1.46x → 1.53x，geo mean 1.39x → 1.44x
三、TMEM 零拷贝操作
6. TMEM A operand for KV MMA (zero-copy v_new)
v_new 直接作为 TMEM A 操作数，避免额外搬运 (7c4a2e6)
7. TMEM A operand for WH MMA (zero-copy h state)
h state 直接驻留 TMEM 作为 WH MMA 的 A 操作数 (d169bfa)
四、寄存器压力优化
8. 寄存器分配调优
CUDA warp 寄存器从 160 → 248：消除 register spilling，spill 指令从 21M → 14M (c8691c7)
最优配置 MMA=64, CUDA=248：性能提升 39% (12272a5)
关键洞察: CUDA wg 寄存器数量影响巨大（248 vs 160 = 39%），MMA 寄存器影响较小
9. 缩短寄存器生命周期
将 output_final_state 移入 else 分支，tTR_rKV 立即释放 (b2740c4)
效果: Local Load Sectors -86.7%，L1 Hit Rate 6% → 83.5%，duration -10.4%
10. 分区 S2R 加载 (Partitioned S2R)
D=128 的 SMEM→RMEM 拆成 2× D=64 half，per-thread 寄存器片段从 64 减到 32 (FP32) (7160426)
效果: H=32 varlen overhead 从 ~30% 降到 ~19%
11. 逐元素 exp(g) 替代批量张量操作
将 bulk .load()/.to()/.exp2() 替换为 element-wise 处理，消除 64-element FP32 SSA vector (012d864)
节省: 每线程峰值节省 ~128-190 个寄存器
12. 持久化 exp2(g) 寄存器跨 Q→K 复用
Q gating 预计算的 exp2(g) 寄存器在 K gating 中复用，避免冗余 G SMEM reload (bb37696)
G SMEM→RMEM 加载次数从 6 → 4
效果: T=8192 non-varlen -6.8%，speedup 1.50x → 1.60x
五、Varlen 优化
13. flashkda 风格 varlen I/O（消除 padding 和 GPU-CPU sync）
消除 output padding、.item() GPU-CPU sync、post-kernel unpadding (1e87d8a)
通过运行时 TMA descriptor 修改处理 tail tiles（copy_tensormap → update_tma_descriptor → fence）
效果: varlen overhead 从 +52% 降到 +27%
14. Persistent kernel for varlen
持久化内核调度，减少 CTA launch 开销和 TMA descriptor 重用 (eb04905)
效果: persistent 1.26x vs non-persistent 1.08x
15. Guard varlen loops with if-partial-chunk
full chunk 走常量路径跳过边界检查，只有 partial chunk 才进入循环 (3b2ac7f)
六、其他有效优化
16. SMEM alignment 从 1024 → 128 字节
SM100 TMA 只需 128B 对齐，节省 ~4.4KB SMEM padding (70fcf4e)
17. Wrapper 层缓存 dummy tensor
缓存未使用的 dummy state/cu_seqlens tensor，消除每次调用 ~0.12ms 的分配开销 (b2a62a4)
短序列因此超过 FLA：T=128 1.23x，T=1024 1.28x

总体效果: 从最初的 0.37x FLA 逐步优化到 non-varlen ~1.60x FLA，varlen ~1.28x FLA。最大的三个跃升来自：TMA 批量搬运（2.1x）、双缓冲流水线（+23-27%）、寄存器调优（+39%）。
