# AI Systems Performance Engineering05--GPU存储I/O优化

**作者**: 想飞的石头带着团队在agi大潮里做挖掘机的

**原文链接**: https://zhuanlan.zhihu.com/p/2043048985087235840

---

​
目录
收起
核心主题
1. 快速存储和数据局部性
1.1 存储带宽需求
1.2 数据局部性原则
2. 顺序vs随机读取模式
2.1 顺序读取优势
2.2 文件组织建议
2.3 随机读取优化
2.4 文件系统优化
3. NVMe和文件系统调优
3.1 I/O调度器
3.2 预读取调优
3.3 NVMe接口优化
3.4 页面缓存
3.5 数据加载器调优
4. NVIDIA GDS (GPUDirect Storage)
4.1 GDS原理
4.2 GDS与GPUDirect RDMA
4.3 GDS支持
4.4 cuFile API
4.5 GDS性能收益
4.6 GDS使用场景
4.7 gdsio基准测试工具
5. DeepSeek的Fire-Flyer文件系统(3FS)
5.1 设计理念
5.2 3FS架构
5.3 3FS特性
5.4 GDS集成注意事项
6. 分布式、并行文件系统和对象存储
6.1 NFS优化
6.2 并行文件系统
6.3 对象存储优化
6.4 数据复制和压缩
7. 监控存储I/O
7.1 监控工具
7.2 PyTorch数据加载分析
7.3 性能诊断
8. 调优数据管道
8.1 高效数据加载
8.2 优化策略
1. 使用多个worker进程/线程
2. 避免Python瓶颈
3. 重叠CPU-GPU处理
4. 批量操作
5. 内存锁定
6. 预取批次
8.3 完整示例
8.4 persistent_workers优势
9. 关键概念总结
9.1 存储优化
9.2 GDS技术
9.3 文件系统
9.4 数据管道
10. 实践建议
10.1 配置检查清单
10.2 性能监控
10.3 故障排查
章节总结
延伸阅读

章节概述

为GPU提供数据与计算本身同样重要。考虑一个在数千个GPU上训练100万亿参数模型的场景，这样的模型可能处理数十亿个训练样本（包括token、图像、音频、视频等）。这意味着必须从存储中读取海量数据并尽快提供给GPU。如果存储管道慢，GPU就会饥饿并闲置。

核心主题
快速存储和数据局部性
顺序vs随机读取模式
NVMe和文件系统调优
NVIDIA GDS (GPUDirect Storage)
DeepSeek的3FS文件系统
分布式存储系统
数据管道调优
1. 快速存储和数据局部性
1.1 存储带宽需求

大规模训练场景：

大型语言模型：数十亿到数万亿训练样本
语言模型：TB级文本数据
视觉模型：PB级图像数据

带宽计算示例：

每个GPU需要: 200 MB/s训练数据
8个GPU总计: 1.6 GB/s聚合带宽

GB200/GB300 NVL72机架:
- 72个Blackwell GPU
- 每GPU需要200 MB/s
- 总计需要: 14-20 GB/s存储吞吐量
1.2 数据局部性原则

最佳实践：

本地存储: NVMe SSD在同一节点
机架本地: NVMe-oF网络拓扑
最小化网络跳数: 提高性能一致性

数据分片策略：

100 TB数据 + 10个节点
→ 每个节点10 TB本地存储
→ 每个节点的数据加载器只读取本地10 TB
→ 避免网络饱和

PyTorch DistributedSampler：

协调worker使每个进程获得唯一数据切片
与跨多个集群节点分片数据的目标一致
2. 顺序vs随机读取模式
2.1 顺序读取优势

GPU偏好：

大型连续块读取
存储对大顺序读取有更高吞吐量

最佳实践：

❌ 避免: 数百万个单独图像文件
   → 大量随机磁盘寻道

✅ 推荐: 少数大型二进制文件
   - Arrow格式
   - TFRecord格式
   - Parquet格式
   - WebDataset tar文件
   - 数据库文件
2.2 文件组织建议

对象存储优化：

Amazon S3: 将小对象合并为大对象
提前准备数据格式

读取块大小调优：

1 MB块 > 4 KB块
原因: 更低的每次读取开销
2.3 随机读取优化

如果必须随机读取：

并行读取:

使用线程调用pread()
Linux异步I/O接口(io_uring)




io_uring优势:

预注册缓冲区
轮询模式
批量提交I/O请求
最小化内核开销
实现高IOPS




2.4 文件系统优化

XFS配置：

# 挂载选项
mount -t xfs -o noatime /dev/nvme0n1 /mnt/data

# noatime: 消除每次读取的访问时间更新

Amazon EFS配置：

使用Max I/O性能模式
从Bursting模式切换到Provisioned吞吐量模式
3. NVMe和文件系统调优
3.1 I/O调度器

现代Linux多队列块I/O调度器：

blk-mq: 跨CPU核心分散I/O

调度器选项：

# 检查当前调度器
cat /sys/block/nvme0n1/queue/scheduler

# 选项:
# - none: 低延迟工作负载标准
# - mq-deadline: 多队列deadline调度器
# - BFQ: Budget Fair Queueing（某些存储设备）

# 设置调度器
echo none > /sys/block/nvme0n1/queue/scheduler

推荐:

高性能NVMe: 使用none或mq-deadline
默认配置通常正确，但值得验证
3.2 预读取调优

内核预读取：

# 查看当前设置
cat /sys/block/nvme0n1/queue/read_ahead_kb
# 默认: 128 KB

# 增加预读取（流式大文件）
blockdev --setra 4096 /dev/nvme0n1
# 设置为4 MB

效果:

减少系统调用开销
流水线读取
提高吞吐量
3.3 NVMe接口优化

PCIe通道:

确保SSD使用最快接口
有足够的PCIe通道
避免瓶颈

RAID 0条带化:

# 多个SSD条带化
# 完全利用设备
# 最大化吞吐量
3.4 页面缓存

Linux页面缓存:

自动缓存最近读取的数据到RAM
中等大小数据集: 热缓存可大幅加速训练
大数据集: 可能超出可用RAM并导致缓存颠簸

预加载到内存:

# 如果数据可放入RAM（包括Grace Blackwell的统一内存）
# 启动时完全预加载到内存
data = load_all_data_to_memory()
# 创建超快的内存缓存
3.5 数据加载器调优

PyTorch DataLoader配置:

DataLoader(
    dataset,
    num_workers=8,           # 多个worker进程
    pin_memory=True,         # 启用内存锁定
    persistent_workers=True, # 持久worker
    prefetch_factor=2,       # 预取因子
)

Worker数量权衡:

太少workers: GPU空闲
太多workers: CPU核心和I/O带宽竞争

目标:
- 磁盘吞吐量接近100%利用率
- CPU有一定余量

高核心CPU（如Grace 72核）:

可以使用更多数据加载worker
注意I/O竞争导致的收益递减
4. NVIDIA GDS (GPUDirect Storage)
4.1 GDS原理

传统路径:

NVMe SSD → CPU内存 → GPU内存
         (复制1)    (复制2)

GDS路径:

NVMe SSD → GPU内存
         (直接DMA)

核心优势:

GPU直接从存储设备读取数据
不在CPU内存中创建额外副本
绕过CPU路径的额外复制
支持本地NVMe和NVMe-oF远程存储
4.2 GDS与GPUDirect RDMA

互补关系:

GDS: 加速存储到GPU的DMA
GPUDirect RDMA: 加速网络到GPU的DMA
两者都移除主机内存弹跳缓冲区
CPU仍负责配置和编排I/O
4.3 GDS支持

硬件要求:

现代NVIDIA GPU
支持直接内存访问的存储栈
正确的NVIDIA驱动和CUDA工具包

支持的存储栈:

本地NVMe和NVMe-oF（XFS/EXT4，使用O_DIRECT）
NFS over RDMA
并行文件系统：
BeeGFS
WekaFS
VAST
IBM Storage Scale
其他集成nvidia-fs的系统




4.4 cuFile API

基本使用:

// 使用cuFile库通过GDS读取文件
cuFileRead(handle, devPtr, size, file_offset, dev_offset);

// 异步API（集成CUDA流）
cuFileReadAsync(handle, devPtr, size, file_offset, dev_offset, stream);
cuFileWriteAsync(handle, devPtr, size, file_offset, dev_offset, stream);

关键配置:

# 使用O_DIRECT启用直接DMA
# 绕过OS页面缓存

# 现代GDS版本支持非O_DIRECT文件描述符
# 但未对齐可能导致额外复制或性能下降
4.5 GDS性能收益

VAST Data报告:

A100 GPU: 20%读取吞吐量提升
H100 GPU: 30%+提升（更高NIC带宽和CPU负担）
图5-1: VAST Data的网络架构（有GDS vs 无GDS）

性能对比:

指标	无GDS	有GDS	提升
吞吐量	8.0 GB/s	9.6 GB/s	+20%
延迟	1.25 ms	1.00 ms	-20%
4.6 GDS使用场景

适合GDS:

CPU饱和处理多个memcpy操作
高I/O速率（如1000 MB/s以上）
数千GPU的大规模训练

不适合GDS:

CPU轻松处理数据传输
可能看不到吞吐量大幅提升
但仍会降低CPU使用率
4.7 gdsio基准测试工具

测试GDS性能:

# CPU路径（无GDS）
/usr/local/cuda/gds/tools/gdsio \
    -f /mnt/data/large_file \
    -d 0 -w 4 -s 10G -i 1M -I 0 -x 2

# GDS路径
/usr/local/cuda/gds/tools/gdsio \
    -f /mnt/data/large_file \
    -d 0 -w 4 -s 10G -i 1M -I 0 -x 0

# 参数说明:
# -f: 文件路径
# -d: GPU设备ID
# -w: worker数量
# -s: 总传输大小
# -i: I/O块大小
# -I: 0=读模式
# -x: 0=GDS路径, 2=CPU路径
5. DeepSeek的Fire-Flyer文件系统(3FS)
5.1 设计理念

问题观察:

AI工作负载执行大量随机读取
传统读取数据缓存对LLM训练和推理无效
甚至适得其反

解决方案:

消除缓存
使用直接文件I/O
每个请求直接到NVMe SSD设备
避免浪费的缓存管理
5.2 3FS架构
图5-2: DeepSeek的Fire-Flyer文件系统(3FS)组件

四大组件:

Cluster Manager: 集群管理器
Metadata Service: 元数据服务
Storage Service: 存储服务
Client: 客户端

互连:

RDMA capable fabric（InfiniBand或RoCE）
最小化CPU参与
避免主机端复制
5.3 3FS特性

关键特点:

Linux文件系统
兼容现有应用
使用RDMA读取进行直接GPU可访问数据传输
元数据分片和复制
数据路径完全绕过OS页面缓存

性能报告:

大规模集群: 7.3 TB/s聚合读取吞吐量

68节点AI-HPC集群:
- 10 × 16 TB NVMe SSDs
- 双100 Gb/s网络
- 聚合读取吞吐量: 6.6 TB/s
- 后台工作负载: 额外1.4 TB/s

对比Ceph: ~1.1 TB/s（类似硬件）
5.4 GDS集成注意事项

FUSE文件系统限制:

FUSE（用户空间）无法提供GDS路径
GDS需要内核级文件系统集成
需要O_DIRECT语义

GDS启用的内核文件系统客户端:

NVMe
NVMe-oF
BeeGFS
WekaFS
IBM Storage Scale
VAST
6. 分布式、并行文件系统和对象存储
6.1 NFS优化

NFS限制:

单个NFS服务器容易成为吞吐量瓶颈
多节点同时读取时

NFS调优:

# 挂载选项
mount -t nfs -o rsize=1048576,wsize=1048576,noatime,async,actimeo=60,lookupcache=pos server:/data /mnt/data

# 参数说明:
# rsize/wsize: 1 MB块大小
# noatime: 消除访问时间更新
# async: 异步写入
# actimeo=60: 缓存文件属性60秒
# lookupcache=pos: 缓存目录条目

适用场景:

少量节点（几节点规模）
大型训练集群应使用并行文件系统
6.2 并行文件系统

Lustre配置:

# 设置文件条带化
lfs setstripe -c 4 /mnt/data/large_file
# -c 4: 跨4个OST条带化

# 性能计算:
# 每个OST: 500 MB/s
# 4个OST: 2 GB/s理论峰值

监控工具:

lmt: Lustre监控工具
厂商特定监控工具
检查存储集群中的热点节点
6.3 对象存储优化

Amazon S3优化:

预暂存数据:

使用s5cmd、aws s3 cp下载到本地NVMe SSD
使用缓存层（如Amazon FSx for Lustre on S3）




流式读取:

使用范围请求
多线程范围Get操作
尽可能大的请求




优化工具:

AWS S3 C++ SDK
s5cmd（多线程工具）




6.4 数据复制和压缩

数据复制:

将数据集复制到每个计算节点
消除网络读取
以额外存储为代价获得立即性能提升

数据压缩:

存储压缩数据（JPEG、Arrow、Parquet）
即时解压缩
节省I/O带宽
消耗额外CPU/GPU周期

GPU解压缩:

nvJPEG: GPU上解码图像
Blackwell Decompression Engine: 支持LZ4、Snappy、Deflate格式
释放SM运行更高价值任务
7. 监控存储I/O
7.1 监控工具

Linux工具:

iostat: I/O统计
iotop: I/O top命令
nvme-cli: NVMe命令行工具
perf: 性能分析
eBPF: 内核跟踪

GPU工具:

Nsight Systems: 跟踪I/O等待时间nsys profile --trace=gds python train.py
DCGM: GPU I/O统计
7.2 PyTorch数据加载分析

测量GPU空闲时间:

import time

start = time.time()
batch = next(data_iterator)
end = time.time()

gpu_idle_time = end - start
# 包括后台预取和H2D复制

分离瓶颈:

# 1. 测量Python管道成本
# 设置num_workers=0，移除后台线程调度
loader = DataLoader(dataset, num_workers=0)

# 2. 测量H2D复制成本
# 使用CUDA事件
start_event = torch.cuda.Event()
end_event = torch.cuda.Event()

start_event.record()
batch_gpu = batch.to("cuda", non_blocking=True)
end_event.record()
torch.cuda.synchronize()
h2d_time = start_event.elapsed_time(end_event)
7.3 性能诊断

典型问题:

GPU等待数据时间: 30%
→ I/O限制GPU吞吐量

优化后:
GPU等待数据时间: 5%
→ 训练速度提升6倍

监控目标:

保持管道充满
从磁盘到GPU内存的每个组件都应监控
确保数据持续流入GPU
8. 调优数据管道
8.1 高效数据加载

典型数据加载流程:

1. 从存储读取数据
2. 解码/反序列化（解析JSON、解码JPEG）
3. 应用转换（tokenize文本、裁剪图像）
4. 整理成批次
8.2 优化策略
1. 使用多个worker进程/线程
DataLoader(
    dataset,
    num_workers=8,  # 每个worker并行获取和预处理数据
)
避免Python GIL问题
主进程异步从worker获取批次
2. 避免Python瓶颈

问题示例:

# ❌ 错误: Python循环逐行tokenize
for line in text_lines:
    tokens = tokenize(line)

解决方案:

# ✅ 正确: 使用优化库
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(text_lines)  # 向量化操作

优化库:

Hugging Face Tokenizers（Rust/C++底层）
TorchText
Python绑定，底层C/C++实现
3. 重叠CPU-GPU处理

理想流水线:

GPU处理批次N:
  CPU已加载和预处理批次N+1
  → 批次N+1在锁定内存中可用
  → GPU完成批次N后立即DMA复制批次N+1
  → CPU继续处理批次N+2
4. 批量操作
# 使用自定义collate_fn批量应用转换
def collate_fn(batch):
    # 批量转换
    return torch.stack([transform(item) for item in batch])

DataLoader(dataset, collate_fn=collate_fn)
5. 内存锁定
DataLoader(
    dataset,
    pin_memory=True,  # 启用内存锁定
)

# 效果:
# - H2D传输更快
# - 允许真正异步的.to(..., non_blocking=True)复制
# - DMA从锁定内存避免额外复制和页面错误

系统配置:

# 设置高ulimit避免大锁定缓冲区分配失败
ulimit -l unlimited

# 或在Docker中
docker run --ulimit memlock=-1:-1 ...
6. 预取批次
loader = DataLoader(
    dataset,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,  # 每个worker预取4个批次
)

预取计算:

预取队列大小 = num_workers × prefetch_factor
= 8 × 4 = 32个批次
8.3 完整示例
import torch
from torch.utils.data import Dataset, DataLoader

class Synthetic(Dataset):
    def __init__(self, n, shape):
        self.n, self.shape = n, shape

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.ones(self.shape, dtype=torch.float32)

B, C, H, W = 32, 3, 224, 224
dataset = Synthetic(n=100_000, shape=(C, H, W))

loader = DataLoader(
    dataset,
    batch_size=B,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

# 使用CUDA流重叠H2D复制和计算
copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.current_stream()

for batch in loader:
    # 在copy_stream上异步传输到GPU
    with torch.cuda.stream(copy_stream):
        batch_gpu = batch.to(device, non_blocking=True)

    # 等待H2D完成
    with torch.cuda.stream(compute_stream):
        torch.cuda.current_stream().wait_stream(copy_stream)
        outputs = model(batch_gpu)
8.4 persistent_workers优势

启用persistent_workers=True:

Worker跨epoch保持活跃
持续填充队列
避免每个epoch边界spawn和tear down进程
对短epoch特别有效
减少模块导入、文件打开等启动开销
9. 关键概念总结
9.1 存储优化
数据局部性: 本地存储 > 机架本地 > 远程存储
顺序读取: 大块连续读取 > 小块随机读取
文件格式: Arrow/TFRecord/Parquet > 单独文件
9.2 GDS技术
直接DMA: 存储到GPU内存，绕过CPU
cuFile API: cuFileRead、cuFileReadAsync
性能提升: 20-30%吞吐量提升
9.3 文件系统
NVMe调度器: none或mq-deadline
预读取: 增加到数MB
并行文件系统: Lustre、GPFS、WekaFS
9.4 数据管道
多worker: num_workers > 0
内存锁定: pin_memory=True
预取: prefetch_factor=2-8
持久worker: persistent_workers=True
10. 实践建议
10.1 配置检查清单
✅ 使用NVMe SSD本地存储
✅ 配置XFS文件系统（noatime）
✅ 设置I/O调度器为none
✅ 增加预读取到4 MB
✅ 启用GDS（如果支持）
✅ 使用多个数据加载worker
✅ 启用内存锁定
✅ 配置预取因子
10.2 性能监控
# 监控NVMe设备
nvme smart-log /dev/nvme0n1

# 监控I/O
iostat -x 1

# 监控GPU利用率
nvidia-smi dmon -s u

# 性能分析
nsys profile --trace=gds python train.py
10.3 故障排查
问题: GPU利用率低
诊断:
1. 检查GPU等待数据时间
2. 监控磁盘吞吐量
3. 分析CPU利用率
4. 检查worker数量

解决:
- 增加worker数量
- 启用pin_memory
- 增加prefetch_factor
- 使用GDS
- 优化数据格式
章节总结

本章详细介绍了GPU存储I/O优化的关键技术：

存储基础: 数据局部性、顺序读取、文件系统调优
GDS技术: 直接DMA、cuFile API、性能收益
3FS文件系统: DeepSeek的专用AI存储系统
分布式存储: NFS优化、并行文件系统、对象存储
数据管道: 多worker、内存锁定、预取、持久worker

这些优化确保存储管道能够跟上GPU的计算速度，避免GPU因等待数据而闲置。

延伸阅读
NVIDIA GPUDirect Storage Documentation
cuFile API Reference
DeepSeek 3FS GitHub Repository
Lustre Documentation
PyTorch DataLoader Documentation
NVIDIA DALI Documentation
