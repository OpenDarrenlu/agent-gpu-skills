# Tensor-001 矩阵乘法分块乘法概述

**原文链接**: https://mp.weixin.qq.com/s/21ztNgVr7sUNu1ajRJfOog

**下载时间**: 2026-06-05 22:38:32

---

GiantPandaLLM

新开一个专题来介绍一下矩阵计算相关的内容，从最基本的算法，到Cutlass这些线性代数模版库，特别是Layout代数相关的内容，后面再逐渐细化到一些硬件实现访存优化和一些算子融合相关的话题，准备工作闲暇时间有点空就补一点，做个长期的专栏。

## 1. GEMM概述

### 1.1 GEMM定义

对于一个矩阵乘法，我们定义如下：

### 1.1.1 内积形式

```
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            C[i][j] += A[i][k] * B[k][j];
```

这种乘法也被称为矩阵乘法的内积形式。随着循环，B矩阵的乘法空间局部性很差，存在多次访问，因此需要缓存一些数据来避免缓存颠簸(cache thrashing)。

### 1.1.2 外积形式

换一种思路，如果我们把K维度放在最外面：

```
for (int k = 0; k < K; ++k) // dim-k at outer loop
    // outer-product for C_i
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] += A[i][k] * B[k][j];
```

这样A和B矩阵都可以按照列和行整个一块的读取。