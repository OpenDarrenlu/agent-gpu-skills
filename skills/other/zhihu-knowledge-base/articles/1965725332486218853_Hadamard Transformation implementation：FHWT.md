# Hadamard Transformation implementation：FHWT

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/1965725332486218853

---

前言

最近看了一些关于rotation quantization 相关的论文，Hadamard Transformation操作频繁出现在论文之中。出于好奇，本文旨在分析如何在GPU上实现高性能的Hadamard Transformation。本期的主题是Tri Dao 23年实现的Fast Hadamard Transform。

前置知识
Hadamard Matrix

Hadamard Matrix 是一个仅有+1和-1组成方阵，其各行互为正交。n维的Hadamard Matrix 与其转置矩阵的乘积的结果为单位矩阵乘上n 即：
𝐻
𝐻
𝑇
=
𝑛
𝐼
,
𝐻
∈
𝑅
𝑛
×
𝑛
 。

对于 
2
𝑛
 维的Hadamard Matrix, 可以通过Sylvester递归的方法利用低维的Hadamard Matrix 通过Kronecker product计算得到，如下：

	
𝐻
1
	
=
[
1
]
,


𝐻
2
	
=
[
1
	
1


1
	
−
1
]
,


𝐻
4
	
=
[
1
	
1
	
1
	
1


1
	
−
1
	
1
	
−
1


1
	
1
	
−
1
	
−
1


1
	
−
1
	
−
1
	
1
]
,

	
 .... 

	
𝐻
2
𝑘
=
[
𝐻
2
𝑘
−
1
	
𝐻
2
𝑘
−
1


𝐻
2
𝑘
−
1
	
−
𝐻
2
𝑘
−
1
]
=
𝐻
2
⊗
𝐻
2
𝑘
−
1
,

通过Sylvester 递归得到的Hadamard Matrix 是对称矩阵，即 
𝐻
=
𝐻
𝑇

Hadamard Transform

(
1
𝑛
𝑋
𝐻
)
(
1
𝑛
𝐻
𝑇
)
=
𝑋
,
𝑋
∈
𝑅
𝑚
×
𝑛
𝐻
∈
𝑅
𝑛
×
𝑛
 ,任意的矩阵X与H的乘积称为Hadamard Transformation，Hadamard Transformation的结果再乘上 
𝐻
𝑇
 得到X。

通过Hadamard Tranformer 可以让原本存在outiler的数值分布变得均匀，从而降低量化的损失。再通过 
𝐻
𝑇
 
(
1
𝑛
𝑋
(
𝑊
1
𝐻
)
)
(
1
𝑛
(
𝐻
𝑇
𝑊
2
)
)
=
𝑋
𝑊
1
𝑊
2
 变换得到原始的值。

QuaRot：https://arxiv.org/pdf/2404.00456
Vectorization for Kronecker product

vec() 把矩阵拍平成一维向量，即 vec(X),X\in\mathbb{R^{n\times n}}, vec(X)\in\mathbb{R^{n n}} 。

对于Kronecker product有以下性质：

vec(LMN) = vec(M)(L ^T \otimes N)

对于Hadamard Transfom,可以推导为(省略缩放值）： XH=X(H_{n1}\otimes H_{n2}),X\in\mathbb{R}^{n\times n},H\in\mathbb{R}^{n\times n},n1\times n2=n\\ vec(X)(H_{n1}\otimes H_{n2})=vec(H_{n1}^T\bar{X}H_{n2}),\bar{X}\in\mathbb{R}^{n\times n1\times n2}

Fast Hadamard Transform Implementation

Fast Walsh–Hadamard transform 通过二分的方法把原本 O(N^2) 计算优化到 O(NlogN) 。

// https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
// FWHT implementation
import math
def fwht(a) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a."""
    assert math.log2(len(a)).is_integer(), "length of a is a power of 2"
    h = 1
    while h < len(a):
        # perform FWHT
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        # normalize and increment
        a /= math.sqrt(2)
        h *= 2

Tri dao 的CUDA 版本的实现仅使用CUDA Core进行计算，每个CTA负责计算 X\in \mathbb{R}^{m\times n} 其中的一行的Hadamard Transform, 即xH,x\in \mathbb{R}^{1\times n} ，通过Thread->Warp->CTA 逐层的迭代计算Hadamard Transform。

以n=2048为例， CTA thread=256， dtype=fp32， 每个thread uint128_t vectorization=4fp32. chunk = 2048/256/4=2，每个thread 拥有reg[2][4]的register本地数据。

核心迭代

xH= x(H_{4}\otimes H_{512})\\ vec(xH)=vec(x(H_{4(\mathbf{4float})}\otimes H_{512})) = vec(H_{4}^Tx'H_{512}), x'\in \mathbb{R}^{4\times 512}\\ =vec(\bar{x'}H_{512}),\bar{x'}=H_{4}^Tx'\in \mathbb{R}^{4\times 512}\\ =vec(\bar{x'}(H_{32(\mathbf{32thread/warp})}\otimes H_{16})) =vec(H_{32}^Tx''H_{16}),x''=\in \mathbb{R}^{4\times 32\times 16}\\ =vec(\bar{x''}H_{16}),\bar{x''}=H_{32}^Tx''\in \mathbb{R}^{4\times 32\times 16}\\ =vec(\bar{x''}(H_{\mathbf{8(8warp/CTA})}\otimes H_{2(\mathbf{2Chunk})})) =vec(H_{8}^Tx'''H_{2}),x'''\in \mathbb{R}^{4\times 32\times 8 \times 2}\\ vec(\bar{x'''}H_{2}),\bar{x'''}=H_{8}^Tx'''\in \mathbb{R}^{4\times 32\times 8 \times 2}

由于CUDA采用Thread->Warp->CTA，三级的线程模型，每一层级的数据访问方式不同，Thread层级数据直接访问register，Warp层级Thread通过shfl访问 ，CTA层级Warp通过Shared memory访问。

overview of FHWT

Thread Level: 对register的thread 独占数据进行变换，即 \bar{x'}=H_{4}^Tx'\in \mathbb{R}^{4\times 512}

Thread Level

Warp Level：对一个Warp内的32 Thread间做变换， \bar{x''}=H_{32}^Tx''\in \mathbb{R}^{4\times 32\times 16}

Warp Level

CTA Level： 对CTA内的8Warp 之间做变换， \bar{x'''}=H_{8}^Tx'''\in \mathbb{R}^{4\times 32\times 8\times 2}

借助Shared Memory 对Warp 之间的数据进行permutation(Transpose），[8,32] ->[32, 8]如下

[t0, t1 ... t31, t32, t31, ... t64, t65,... t254, t255] -> [t0 t32 t96 ... t224 t1 t33 t96...t225 .... t31 t95 ...255] (代表变换前后，连续的256 个thread,拥有的对于原始thread的数据 )通过smem 让连续的8个thread 存储8个warp对于 interleave的数据， 再通过warp level 只变换前八个 thread 的数据，得到 CTA level的变换结果。

再通过smem 把thread的数据permutation 回原始的数据排布

[t0 t32 t96 ... t224 t1 t33 t96...t225 .... t31 t95 ...255] ->[t0, t1 ... t31, t32, t31, ... t64, t65,... t254, t255]

smem permutation

Chunk Level：对每个Thread 的Chunk 之间进行变换 \bar{x'''}H_2\in \mathbb{R}^{4\times 32\times 8 \times 2}

最后需要对每个Thread Chunk 维度之间进行变换，对于 Chunk=2^N 直接在chunk维度调用thread_level的变换。其外对于Chunk=12/20/28/40等值，调用对应的特化非二次幂的Hadamard Transfom。https://neilsloane.com/hadamard/ 可以在这里查到一些非二次幂的Hadamard Matrix。

总结

最开始看FHWT实现的时候，对不同Thread层级之间的操作一头雾水，后来把不同Thread层级的操作通过Vectorization for Kronecker product拆解，把n维度的行向量拆解到不同并行层级，就比较好理解不同层级之间操作的意义。也感谢周老师 @ChiveArchitect 给的一些交流和指导。最后Tri Dao yyds。
