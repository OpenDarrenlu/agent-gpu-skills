# Triton极简入门: Triton Vector Add

**原文链接**: http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247527533&idx=1&sn=6a186f514e42e543c0d3ff3567ff7656

**下载时间**: 2026-06-05 22:16:50

---

作者丨DefTruth
来源丨https://zhuanlan.zhihu.com/p/1902778199261291694
编辑丨GiantPandaLLM

### 0x00 前言

后续会陆续更新一些CUDA和Triton Kernel编程入门向的文章。

### 0x01 Triton编程基础

核心点：Triton的编程粒度是Block（每个Block只会被调度到一个SM上），而不是Thread。我们只需要考虑每个Block需要做什么，至于Thread/Warp的分布和调度，Triton自动给我们处理了。

传统的基于 CUDA 进行 GPU 编程难度较大，在优化 CUDA 代码时，必须考虑到数据流在DRAM、SRAM 和 ALU之间的Load/Store的问题，还需要仔细考虑到Grid、Block、Thread和Warp等不同级别的调度优化问题。

Triton 的出现，降低了CUDA Kernel编写的难度，它将一些需要精心设计的优化策略进行自动化，比如内存事务合并、SRAM分配和管理、流水线优化等，从而使得编程人员可以将更多的精力放在算法本身。

### 0x02 Triton Vector Add

通过add_kernel示例讲解Triton kernel的编程方式：
- 使用tl.program_id(axis=0)获取当前program id
- 使用block_start=pid*BLOCK_SIZE和offsets=block_start+tl.arange(0,BLOCK_SIZE)计算数据偏移
- 使用mask=offsets<n_elements创建mask防止越界
- 使用tl.load和tl.store进行数据加载和写入

### 0x03 PyTorch封装

Triton将会传入的Tensor当成指针来处理，而非数据张量。kernel启动时，只需要考虑一个grid中block的布局。

### 0x04 PTX Gen code

通过指定TRITON_CACHE_DIR环境变量，可以保存Triton生成的中间IR文件进行分析。关注PTX文件，看是否使用向量化访存指令。

本案例生成的PTX使用了ld.global.v4.b32和st.global.v4.b32这两个向量化访存的指令。

### 0x05 性能

Triton Vector Add Kernel和pytorch的add算子，性能基本一致。

### 0x06 总结

本文简单对比了Triton Kernel编程和CUDA编程的主要区别，说明了 Triton的编程粒度是Block，而不是Thread；介绍了通过PTX分析Gen code的方式；通过Vector Add的示例来讲解Triton kernel的编程方式。

代码在：https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/openai-triton/elementwise