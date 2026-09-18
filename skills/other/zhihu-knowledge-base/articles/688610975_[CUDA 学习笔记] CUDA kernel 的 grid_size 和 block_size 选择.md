# [CUDA 学习笔记] CUDA kernel 的 grid_size 和 block_size 选择

**作者**: PicoPika一入infra深似海

**原文链接**: https://zhuanlan.zhihu.com/p/688610975

---

CUDA kernel 的 grid_size 和 block_size 选择
核函数执行配置

Execution Configuration

cuda_kernel<<< Dg, Db, Ns, S >>>(...)

Dg: grid 的维度和大小 (grid_size). 类型 dim3. : Dg.x * Dg.y * Dg.z 为启动的线程块(block)数.
Db: 每个线程块的维度和大小 (block_size). 类型 dim3. Db.x * Db.y * Db.z 为每个线程块的线程数.
Ns: 每个线程块需动态分配的共享内存的字节数. 类型 size_t. 默认值 0.
S: 指定的相关联的 CUDA 流. 类型 cudaStream_t. 默认值 0.
block_size 选择

NVIDIA GPU 算力及规格参数

大于 0, 上限为 1024
x 维度和 y 维度上限 1024, z 维度上限 64
最好是 32 的倍数. 因为一个 warp 有 32 个线程.

最好是不小于 SM 上最大同时执行的线程数(Maximum number of resident threads per SM)和最大同时执行的线程块数(Maximum number of resident blocks per SM)的比值.
因为要尽可能让 GPU 占有率(Occupancy, SM 上并发执行的线程数和 SM 上最大支持的线程数的比值)达到 100%, 所以:

理
论
线
程
数
最
大
数
最
大
线
程
数
最
大
线
程
数
最
大
数
	
𝑆
𝑀
理
论
线
程
数
=
𝑏
𝑙
𝑜
𝑐
𝑘
_
𝑠
𝑖
𝑧
𝑒
×
𝑆
𝑀
最
大
𝑏
𝑙
𝑜
𝑐
𝑘
数
≥
𝑆
𝑀
最
大
线
程
数

	
⇒
 
𝑏
𝑙
𝑜
𝑐
𝑘
_
𝑠
𝑖
𝑧
𝑒
≥
𝑆
𝑀
最
大
线
程
数
/
𝑆
𝑀
最
大
𝑏
𝑙
𝑜
𝑐
𝑘
数

V100 、 A100、 GTX 1080 Ti 为 2048 / 32 = 64, RTX 3090 为 1536 / 16 = 96. 因此 block_size 不应小于 96.

最好是 SM 最大线程数的约数(因数). 因为 block 调度到 SM 的过程是原子的, 即该 block 中的所有线程都在同一 SM 上执行, 因此 block_size 为 SM 最大线程数的约数时可以确保 SM 不会有一直空闲的线程.
主流架构最大线程数(2048, 1536, 1024)的公约数为 512, 256, 128.

寄存器、共享内存等资源对应到每个线程不能超过上限(每个 block 的 32 位寄存器数量, 每个 block 的共享内存大小上限). 这里指明为"对应到每个线程", 即每个线程所使用的寄存器数、共享内存应小于上限/ block_size.


在不考虑线程同步等其他因素的情况下:

当寄存器和共享内存使用较少时, 可以将 block_size 设置为较大的 512、1024(SM 最大线程数不为 1536 时);
反之, 当寄存器和共享内存使用较多时, 可以将 block_size 设置的较小, 即 128、256.
※ 在笔者接触的一些 CUDA 库中, block_size 一般多被设置为 128、256
grid_size 选择
x 维度上限 2^{31}-1 , y 维度和 z 维度上限 65535.
不应低于 GPU 上 SM 的数量 (A100 为 108 个). 因为至少让每个 SM 都启动 1 个 block
最好不低于 SM数\times 每个SM最大block数 . 这样一批 GPU 可以一次几乎同时完成的 block 称之为 wave, 这里的"每个 SM 最大 block 数"根据实际情况会不同.
数量足够多的整数个 wave, 或数量足够大. 考虑到 GPU 的多 CUDA 流等情况, 仍可能出现尾效应(tail effect, 一个 wave 完成后, GPU 上将只有很少的 block 在执行), 因此 grid_size 足够大可以让 GPU 尽可能充分调度运行.

如下是 Oneflow 中的计算方式:

unsigned grid_size = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));

n: 数据个数
kBlockSize: block_size
sm_count: SM 个数
tpm: SM 上最大同时执行的线程数(Maximum number of resident blocks per SM)
kNumWaves: wave 个数(上文有提到), 一般设置为 32. 使 grid 为整数个 wave.

数据量较小的情况下, 不会启动过多的线程块 ((n + kBlockSize - 1) / kBlockSize); 在数据量较大的情况下, 尽可能将线程块数目设置为数量足够多的整数个 wave, 以保证 GPU 实际利用率够高 (sm_count * tpm / kBlockSize * kNumWaves).

参考资料
如何设置CUDA Kernel中的grid_size和block_size？ - 知乎
高效、易用、可拓展我全都要：OneFlow CUDA Elementwise 模板库的设计优化思路 - 知乎
