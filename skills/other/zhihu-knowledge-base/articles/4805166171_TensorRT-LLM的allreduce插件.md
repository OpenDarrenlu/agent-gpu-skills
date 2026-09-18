# TensorRT-LLM的allreduce插件

**作者**: yylloc以太网超节点

**原文链接**: https://zhuanlan.zhihu.com/p/4805166171

---

周末英伟达发了篇博客，讲TensorRT-LLM如何绕开NCCL优化单机多卡allreduce。

https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/

这是TensorRT-LLM集合通信插件的调用图；有背景色的是算子，其它是host侧调用。

图片在https://github.com/YconquestY/cc；如果觉得有帮助，麻烦给个star

这个插件最早可以追溯到FasterTransformer。在TensorRT-LLM中，插件主要定义在cpp/tensorrt_llm/kernels/customAllReduce.cu。它在10月8号加入lamport_style_one_shot_all_reduce_norm…相关内容（上图粉色），利用NVSwitch multicast特性加速allreudce；这也是博文的重要内容。此外，下半年比较重要的更新还有

在7月23号加入FDL（后更名为PDL）相关算子启动调用。不清楚FDL/PDL是啥的缩写，但按照读NCCL源码的经验，它主要面向Hopper及更新的设备（__CUDA_ARCH__ >= 900）
在7月4号更新multi-gpu_barrier和block_barrier；单机多卡barrier在下文提到过
yylloc：手撸算子是人力密集型劳动…？
207 赞同 · 23 评论 文章

这个插件的多卡barrier运用PTX memory order很娴熟，写出来功力不低…例如

static inline __device__ void st_flag_release(uint32_t const& flag, uint32_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
    __threadfence_system();
    asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

PTX memory consistency玩儿得更溜的是NCCL和NVSHMEM。强烈建议读一下NVSHMEM中asm PTX带weak memory order的部分，很精彩。 PTX memory consistency是什么？英伟达发过2篇文章：

当然，文章对理论（尤其是形式化验证）的功底要求很高，没听过的话就不要读了，读不太懂的 :( 倒是可以借鉴下 @kaitoukito 的

这里面verification的东西很多，知道咋用就行了，解释原理三四篇文章都说不清楚 :P

然后不要被英伟达博客的MultiShot吓到，shot不是啥新词，换成pass、turn也行，翻译过来就是轮 。oneShotAllReduceKernel/…one_shot_all_reduce…是多卡传1轮数据（加当前卡计算）能完成allreduce运算，twoShotAllReduce就是多卡得传2轮（加计算）才能完成。

最后吐槽一下TensorRT-LLM的工程…很难想象customAllReudce.cu单个文件就有驼峰和下划线2种命名方法，看得非常难受。

催更 @heyguy，很期待你的
