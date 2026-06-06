# [深入分析CUTLASS系列] 0x04 cutlass 源码分析(三) ---  多级流水线(software pipeline)

**作者**: JoeNomad​​新南威尔士大学 信息技术硕士

**原文链接**: https://zhuanlan.zhihu.com/p/687397095

---

​
目录
收起
开篇
Prelogue
多级流水线概念
gpu的多级memory
MAIN
Warp中gemm逻辑
处理数据依赖
Epilogue
TVM中software pipeline pass
开篇

大家好，我是Joe，本文篇幅将会比较短，是源码分析的最后一篇，会展开讲讲多级流水线，其他优化手段前面几篇文章已经介绍的差不多了。后面的话会讲讲更high level的东西，如在其他AI infra中应用cutlass。

Prelogue
多级流水线概念

多级流水线这个概念历史已经非常久远了，想要完成的事情是overlap不同的硬件单元处理的事物，希望提高整体的硬件利用率。在理想的情况下，我们希望在程序执行时，每个需要用到的硬件都是忙碌的，尽量避免等待。

instruction pipeline (from wiki)
gpu的多级memory

smem的数据搬运: global->L2->L1->RF->smem

cp.async会bypass L1和RF, 把数据直接从L2 copy 到smem中

gpu memory hierarchy
MAIN
Warp中gemm逻辑

在Multi-level-tiling的GEMM计算逻辑中，不启用splitk算法的情况下，每个warp会处理一个被切分的块(tile_m, k) x (k, tile_n)，由于shared_memory的限制，我们会tile k维度，于是计算逻辑就会变成循环k/tile_k次，得到小块的最终计算值

single pipeline code
处理数据依赖

for循环中由于我们计算的是一个reduce sum，所以最终的累加部分是有读写依赖的，但是当我们拆开来看，累加之前的部分每个iter是相互独立的。以stage2为例，我们可以用 2 x smem + 2 x register file (ldg->smem->rf->mma->rf)，来保证两个不同iter中在累加之前没有数据依赖问题。推广到多级流水线，我们可以得出, RF和smem的用量和stage数是线性关系。

overlap示意图

多条流水线需要通过异步拷贝cp.async这个asm来完成, PTX参考：

cutlass中copy async的实现

sm80之前的硬件只能启用两条流水线，逻辑也比较简单，处理好数据依赖，在循环展开的时候编译器会自己完成load compute并行。异步多流水在逻辑并没有太大的区别，只是在计算当前iter需要用到的smem,rf的下标会有所不同。

mma_pipelined.h gemm compute 入口
Epilogue
TVM中software pipeline pass

在tvm中，injective_software_pipeline也做了相同的事情，在tvm script中对for loop加上annotation，在执行该pass的时候就会把循环展开并完成相应的数据依赖处理的变换。如何完成的细节可以参考http://inject_software_pipeline.cc这个文件，sample如下：

# before
@T.prim_func
def simple_compute(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]):
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(
            0,
            16,
            annotations={
                "software_pipeline_stage": [0, num_stages],
                "software_pipeline_order": [0, 1],
            },
        ):
            with T.block("compute"):
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)

# after
@T.prim_func
def transformed_simple_compute(
    A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]
) -> None:
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0:16]])
            T.writes([C[tx, 0:16]])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0]])
                T.writes([B[0, tx, 0]])
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with T.block():
                T.reads([A[tx, 1:16], B[0:2, tx, 0]])
                T.writes([B[0:2, tx, 0], C[tx, 0:15]])
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1]])
                        T.writes([B[(i + 1) % 2, tx, 0]])
                        B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with T.block():
                        T.reads([B[i % 2, tx, 0]])
                        T.writes([C[tx, i]])
                        C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
            with T.block():
                T.reads([B[1, tx, 0]])
                T.writes([C[tx, 15]])
                C[tx, 15] = B[1, tx, 0] + T.float32(1)





Cutlass相关的优化手段至此应该已经写完了，如有疏漏||错误还麻烦大家评论指出~

感谢大家阅读

相关内容导览:
