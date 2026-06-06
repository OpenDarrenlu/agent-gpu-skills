# AI Systems Performance Engineering06--GPU架构、CUDA编程和最大化占用率

**作者**: 想飞的石头带着团队在agi大潮里做挖掘机的

**原文链接**: https://zhuanlan.zhihu.com/p/2043054114439959622

---

章节概述

本章回顾单指令多线程(SIMT)执行模型，以及warp、线程块和网格如何将GPU算法映射到流式多处理器(SM)。我们将深入CUDA编程模式，讨论片上内存层次结构，并介绍roofline分析来识别计算密集型vs内存密集型内核。

核心主题
SIMT执行模型
GPU架构和SM结构
CUDA线程层次结构
内存层次结构
CUDA编程模式
Roofline分析
1. 理解GPU架构
1.1 GPU vs CPU设计理念

CPU设计:

优化低延迟单线程性能
少量强大核心
复杂控制逻辑
大缓存

GPU设计:

优化吞吐量的处理器
运行数千个并行线程
大量简单核心
简单控制逻辑
高并行度
1.2 CUDA编程流程
图6-1: 简单的CUDA编程流程
1. 主机加载数据到CPU内存
2. 从CPU复制数据到GPU内存
3. 调用GPU内核处理GPU内存中的数据
4. CPU将结果从GPU内存复制回CPU内存
5. 结果回到CPU进行进一步处理
1.3 流式多处理器(SM)

SM特性:

类似CPU核心，但优化吞吐量
每个SM可跟踪最多64个warp（Blackwell）
每个SM最多2048个线程并发

Blackwell SM资源:

每个SM:
- 64K个32位寄存器 (256 KB)
- 256 KB L1缓存/共享内存
- 最多227 KB可配置为用户管理的共享内存
- 单个线程块最多请求227 KB动态共享内存
1.4 Warp调度器

Blackwell SM架构:

4个独立的warp调度器
每个调度器可每个周期发射一个warp指令
支持双发射：一个算术指令 + 一个内存指令
图6-2: Blackwell SM包含四个独立的warp调度器




调度能力（表6-1）:

指标	数值
调度器数量	4个
最大发射warp数	4个（每个调度器一个）
最大数学操作	4个（每个调度器的算术发射）
最大内存操作	4个（每个调度器的加载/存储发射）

最佳情况:

每个周期可双发射4个数学和4个内存指令
跨4个warp
最大化计算和内存吞吐量
1.5 特殊功能单元(SFU)

功能:

处理超越函数操作（sin、cos、倒数、平方根）
使用专用SFU流水线
独立于主INT32/FP32和加载/存储流水线

优势:

SM可继续发射数学和内存指令
不等待较慢函数完成
增加指令级并行性
2. 线程、Warp、块和网格
2.1 CUDA线程层次结构

三级层次:

线程(Thread): 执行内核代码
线程块(Thread Block/CTA): 最多1024个线程
网格(Grid): 线程块组成网格




图6-3: 线程、线程块和网格




图6-4: 线程层次结构视图（包括CPU主机）




2.2 线程块集群(Thread Block Clusters)

现代GPU特性:

不同线程块的线程可以相互协作
跨SM通信
访问彼此的共享内存
使用硬件支持的集群范围屏障
图6-5: 线程块集群中使用的硬件支持DSMEM

分布式共享内存(DSMEM):

硬件特性
通过快速片上互连链接线程块集群中所有SM的共享内存库
多SM分布式共享内存池
允许不同块的线程以片上速度读取、写入和原子更新彼此的共享缓冲区
不使用全局内存带宽
2.3 Warp执行模型

Warp定义:

32个线程为一组
在SIMT模型下锁步执行
由warp调度器管理
图6-7: Warp（32线程）作为一个整体推进

高占用率:

保持更多warp在执行中
当一个warp停顿时，另一个准备运行
保持GPU计算单元忙碌

平衡:

高占用率 vs 每线程资源限制
寄存器和共享内存限制
寄存器溢出到较慢内存会创建新停顿
2.4 Warp分歧(Warp Divergence)

问题:

Warp中的线程必须遵循相同的控制流路径
如果某些线程走if路径，其他走else路径
Warp序列化执行，顺序处理每个分支路径
图6-8: SIMT warp分歧（左）vs 统一性（右）

性能影响:

通过屏蔽非活动通道
运行额外遍历覆盖每个分支
执行时间乘以分支数量

重要说明:

分歧只影响单个warp内的线程
不同warp可以遵循不同分支，无性能损失
3. 选择线程块和网格大小
3.1 线程块大小选择

关键原则:

与硬件的32线程warp大小对齐
选择32的倍数作为线程块大小

示例:

256线程块 = 8个warp = 256 ÷ 32
✓ 完全占用每个warp

33线程块 = 2个warp槽
✗ 第二个warp只使用1/32的通道
✗ 浪费并行机会
3.2 硬件限制

Blackwell B200限制（表6-2）:

资源	硬件限制	说明
Warp大小	32线程	基本SIMT执行单元
每线程块最大线程数	1,024	blockDim.x × blockDim.y × blockDim.z ≤ 1024
每线程块最大warp数	32	1,024 ÷ 32 = 32

SM驻留限制（表6-3）:

资源（每SM）	硬件限制	说明
每SM最大驻留warp	64	64 × 32 = 2,048线程
每SM最大驻留线程	2,048	等于64 warp × 32线程/warp
每SM最大活动块	32	硬件限制
3.3 占用率计算

示例:

每个块1,024线程:
- 每SM最多2个块 (64 warp)
- 2,048线程

每个块256线程:
- 每SM最多8个块 (64 warp)
- 2,048线程
- 可能增加占用率，帮助隐藏延迟

权衡:

太大的块: 需要太多寄存器/共享内存 → 溢出
太小的块: 调度开销增加
4. GPU内存层次结构
4.1 内存层次

从快到慢:

寄存器 (最快, 每SM 256 KB)
    ↓
共享内存/L1缓存 (每SM 256 KB)
    ↓
L2缓存 (126 MB, Blackwell)
    ↓
HBM3e (192 GB, 8 TB/s带宽)
4.2 寄存器文件

特性:

最快的存储
每线程私有
Blackwell: 每SM 64K个32位寄存器

限制:

每线程最多255个寄存器
寄存器使用影响占用率
4.3 共享内存

特性:

片上SRAM
用户管理
低延迟
线程块内线程共享

Blackwell配置:

每SM 256 KB L1/共享内存
最多227 KB可配置为共享内存
可动态分配

使用场景:

缓存频繁访问的数据
线程间数据共享
减少全局内存访问
4.4 L2缓存

特性:

所有SM共享
较大容量
缓存全局内存访问

Blackwell:

126 MB L2缓存
每芯片63 MB
4.5 HBM3e内存

特性:

高带宽内存
GPU主内存
大容量

Blackwell B200:

192 GB容量（180 GB可用）
8 TB/s带宽
5. CUDA编程模式
5.1 基本内核结构
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 启动内核
int blockSize = 256;
int numBlocks = (n + blockSize - 1) / blockSize;
vectorAdd<<<numBlocks, blockSize>>>(a, b, c, n);
5.2 内存管理

主机到设备复制:

cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

设备到主机复制:

cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

异步复制:

cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
5.3 共享内存使用
__global__ void matMul(float *A, float *B, float *C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 加载到共享内存
        sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();

        // 计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}
6. Roofline分析
6.1 Roofline模型

目的:

识别内核是计算密集型还是内存密集型
可视化性能瓶颈

关键指标:

算术强度(Arithmetic Intensity): FLOPS/Byte
峰值FLOPS: GPU最大计算能力
内存带宽: GPU最大内存吞吐量
6.2 Roofline图
性能 (GFLOPS)
    ^
    |           计算上限
    |          /
    |         /
    |        /  内存上限
    |       /  /
    |      /  /
    |     /  /
    |    /  /
    |___/__/________________> 算术强度 (FLOPS/Byte)
6.3 内核分类

内存密集型:

算术强度低
受内存带宽限制
优化重点: 减少内存访问、增加数据重用

计算密集型:

算术强度高
受计算能力限制
优化重点: 向量化、并行化
6.4 优化策略

内存密集型内核:

使用共享内存缓存
合并内存访问
减少内存流量

计算密集型内核:

使用Tensor Cores
循环展开
指令级并行
7. Tensor Memory Accelerator (TMA)
7.1 TMA功能

定义:

专用硬件单元
异步数据传输
张量内存(TMEM)作为Tensor Core操作的累加器

优势:

减少寄存器压力
异步加载多维张量
与计算重叠
7.2 TMA使用
// 异步拷贝到共享内存
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);

// 使用TMA加载张量
// (具体API取决于CUDA版本)
8. 关键概念总结
8.1 GPU架构
SM: 流式多处理器，GPU核心计算单元
Warp: 32线程，SIMT执行单元
占用率: SM上活跃warp的比例
8.2 内存层次
寄存器: 最快，每线程私有
共享内存: 片上SRAM，线程块共享
L2缓存: 所有SM共享
HBM: 高带宽主内存
8.3 编程模式
线程层次: 线程 → 线程块 → 网格
内存管理: cudaMemcpy、cudaMemcpyAsync
同步: __syncthreads()
8.4 性能分析
Roofline模型: 识别瓶颈
算术强度: FLOPS/Byte
内存密集型 vs 计算密集型
9. 实践建议
9.1 线程块大小选择
✅ 使用32的倍数
✅ 常见选择: 128、256、512
✅ 考虑寄存器和共享内存限制
✅ 使用CUDA Occupancy Calculator
9.2 内存优化
✅ 使用共享内存缓存频繁访问的数据
✅ 合并全局内存访问
✅ 避免寄存器溢出
✅ 使用异步内存操作
9.3 性能分析
# 使用Nsight Compute分析内核
ncu --set full ./my_kernel

# 使用Nsight Systems分析整体性能
nsys profile ./my_application

# 检查占用率
ncu --metrics sm__warps_active.avg.pct_of_peak ./my_kernel
章节总结

本章详细介绍了GPU架构和CUDA编程基础：

GPU架构: SM、warp调度器、内存层次
线程层次: 线程、warp、线程块、网格
内存管理: 寄存器、共享内存、L2、HBM
编程模式: 内核编写、内存操作、同步
性能分析: Roofline模型、算术强度

这些基础知识为后续章节的高级优化技术奠定了基础。

延伸阅读
NVIDIA CUDA Programming Guide
NVIDIA CUDA Best Practices Guide
NVIDIA Nsight Compute Documentation
NVIDIA Nsight Systems Documentation
CUDA Occupancy Calculator
