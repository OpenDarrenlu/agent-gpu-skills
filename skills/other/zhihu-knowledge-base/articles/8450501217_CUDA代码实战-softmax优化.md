# CUDA代码实战-softmax优化

**作者**: 李安渝infra小菜鸡

**原文链接**: https://zhuanlan.zhihu.com/p/8450501217

---

​
目录
收起
softmax 算子原理
saft softmax 原理及python实现
online softmax原理及python实现
cuda优化
基础baseline
使用shared memory
擅用warp结构
使用shared memory + warp
优化online softmax
参考
softmax 算子原理

Softmax 是一种基本的激活函数，是基于 sigmoid 二分类函数在多分类任务上的推广，它可以将一个数值向量归一化为一个概率分布向量，且各个概率之和为1。在多分类网络中，常用 Softmax 作为最后一层进行分类。

softmax 在 flash attention中占据重要地位，掌握了softmax几种变体的cuda优化，就可以离优化flash attention 迈进巨大的一步。

在这里推荐一篇写的非常易懂的手稿From online softmax to flash attention。接下来，让我们正式开始介绍softmax这个非常重要的算子。

softmax的数学公式为：

𝑠
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
𝑥
𝑖
)
=
𝑒
𝑥
𝑖
∑
𝑗
𝑁
𝑒
𝑥
𝑗

手撕代码实现-python版本：

import torch 
print('torch 手撕')
A = torch.tensor([[-0.3, 0.2, 0.5, 0.7, 0.1, 0.8]])
A_exp = torch.exp(A)
A_sum = torch.sum(A_exp, dim=1).unsqueeze(1)
P = A_exp / A_sum #广播
print(P)
#结果
tensor([[0.0827, 0.1364, 0.1841, 0.2249, 0.1234, 0.2485]])
saft softmax 原理及python实现

基于传统的softmax会有一个非常扎眼的问题，就是在当输入的数据中有变量和其他值 差距非常明显时softmax输出的结果将会发生数值溢出现象：

上溢：数值较大的数据经过一些运算后其数值非常大，以至于超过计算机的存储范围而无法继续运算，在程序中表现为 NAN
下溢：非常接近0 的数据被四舍五入为 0，从而产生毁灭性的舌入误差。
A = torch.tensor([[10000, 0.2, 0.5, 0.7, 0.1, 0.8]])

#结果
tensor([[nan, 0., 0., 0., 0., 0.]])

所以为了解决softmax中因为e的次幂运算带来的数值溢出，safe softmax被提出来

softmax(x_i) = safe\_softmax(x_i)= \frac{e^{x_i-max(x)}}{\sum^N_j( e^{x_j-max(x)})}

import torch 
print('torch 手撕')
A = torch.tensor([[10, 12, 12, 12, 12, 12]])
A_max = A.max()
A_exp = torch.exp(A-A_max) 
A_sum = torch.sum(A_exp, dim=1).unsqueeze(1)
P = A_exp / A_sum #广播
print(P)
#结果 
torch 手撕
tensor([[0.0264, 0.1947, 0.1947, 0.1947, 0.1947, 0.1947]])


c++版本

void softmax_forward_cpu(float* out, const float* inp, int N, int C){
    // 输入数据大小是（N，C）
    for(int i =0; i< N;i++){
        // 计算每一行（每一类的
        float max_val = INT_MIN;
         //1、获取输入的最大值
         // 获取输入数据位置的指针地址 
        for(int j=0;j<C;j++){
            int index = i*N+j;
            if(inp[index]>max_val)max_val=inp[index];
        }
        // 第二步计算 sum值
          double sum = 0.0;
        for(int j=0;j<C;j++){
            int index = i*N+j;
            sum+=expf(inp[index]-max_val);
        }
        // 根据 这个计算 最后答案
        float norm = 1.f / (float)sum;
        for(int j=0;j<C;j++){
            int index = i*N+j;
            out[index]=expf(inp[index]-max_val)*norm;
        }
    }
}

online softmax原理及python实现

介绍完safe softmax的基本原理，我们将safe softmax 的每个步骤拆分来看，

\begin{array}{l} for \ \ \ \ i \longleftarrow 1,N \ do \\ \ \ \ \ \ \ \ \ m_i \longleftarrow max(m_{i-1},x_i) \\ end \\ for \ \ \ \ i \longleftarrow 1,N \ do \\ \ \ \ \ \ \ \ \ sum_i \longleftarrow sum_{i-1} + e^{x_i-m_N} \\ end \\ for \ \ \ \ i \longleftarrow 1,N \ do \\ \ \ \ \ \ \ \ \ a_i \longleftarrow \frac{e^{x_i-m_N}}{sumn_N} \\ end \\ \end{array}

可以看到，safe softmax最关键的步骤其实是两个reduce操作（max，sum），这样会反复对内存进行读写操作 ，而online softmax则将这两个reduce合并在一个循环里面，从而显著减少内存的IO次数，以大幅度提高执行效率

\begin{array}{l} for \ \ \ \ i \longleftarrow 1,N \ do \\ \ \ \ \ \ \ \ \ m_i \longleftarrow max(m_{i-1},x_i) \\ \ \ \ \ \ \ \ \ sum_i^{'} \longleftarrow sum_{i-1}^{'}e^{m_{i-1}-m_i}+ e^{x_{i}-m_i}\\ end \\ for \ \ \ \ i \longleftarrow 1,N \ do \\ \ \ \ \ \ \ \ \ a_i \longleftarrow \frac{e^{x_i-m_N}}{sumn_N} \\ end \\ \end{array}

接下来，我们推导一下这个变化的由来 ，可以看到原来的safe softmax 的sum计算是要依赖于max操作得到的最大值的，所以我们需要改造 sum_i \longleftarrow sum_{i-1} + e^{x_i-m_N} 摆脱对 m_N 的依赖，从而将第一个循环和第二个循环融合起来，达到减少IO次数的目的。

\begin{array}{l} sum_N = \sum_{i}^{N}e^{x_i-m_N} \\ sum_{N+1} = \sum_{i}^{N+1}e^{x_i-m_{N+1}} \\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = \sum_{i}^{N}e^{x_i-m_{N+1}} + e^{x_{N+1}-m_{N+1}}\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = \sum_{i}^{N}e^{x_i-m_{N}}e^{m_{N}-m_{N+1}} + e^{x_{N+1}-m_{N+1}}\\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = sum_Ne^{m_{N}-m_{N+1}}+ e^{x_{N+1}-m_{N+1}} \end{array}

可以看到，经过公式推导sum运算已经可以不再依赖全局最大值了，而是可以依赖到目前为止的最大值进行sum运算 ，从而将max操作和sum操作放进同一个循环，减少IO的频繁操作。

为了后续的cuda的代码对比，这里直接写上c++版本

// online softmax 实现->2 -pass 
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C){
    // 输入数据大小是（N，C）
    for(int i =0; i< N;i++){
        // 计算每一行（每一类的
        float max_val = INT_MIN;
        double sum = 0.0;
         //1、获取输入的最大值
         // 获取输入数据位置的指针地址 
        for(int j=0;j<C;j++){
            int index = i*N+j;
            float maxval_prev = max_val;
            if(inp[index]>max_val){
                // 修改
                max_val=inp[index];
                sum = sum*expf(maxval_prev - max_val) + expf(inp[index]-max_val);
            } else {
                sum += expf(inp[index]-max_val);
            }
        }
        // 第二步计算 sum值
        // 根据 这个计算 最后答案
        float norm = 1.f / (float)sum;
        for(int j=0;j<C;j++){
            int index = i*N+j;
            out[index]=inp[index]*norm;
        }
    }


}

cuda优化

介绍完softmax和其变体的原理和cpu版本代码的介绍，我们现在开始介绍其cuda上的优化。上文有介绍，softmax家族最关键的两个操作都是属于reduce操作（sum、max)，所以我们还是采用对reduce算子的优化，接着优化咱们的softmax。（具体的reduce优化介绍，可以参考我的上一篇文章

基础baseline

只是通过gpu高并发的特性来完成softmax

// gpu实现
__global__  void softmax_forward_gpu(float* out, const float* inp, int N, int C){
    // 每个线程负责一个元素的计算，也就是取代外层N的循环，在N维上并行起来
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<N){
       // 计算每一行（每一类的
        float max_val = INT_MIN;
         //1、获取输入的最大值
         // 获取输入数据位置的指针地址 
        for(int j=0;j<C;j++){
            int index = i*N+j;
            if(inp[index]>max_val)max_val=inp[index];
        }
        // 第二步计算 sum值
          double sum = 0.0;
        for(int j=0;j<C;j++){
            int index = i*N+j;
            sum+=expf(inp[index]-max_val);
        }
        
        // 根据 这个计算 最后答案
        float norm = 1.f / (float)sum;
        for(int j=0;j<C;j++){
            int index = i*N+j;
            out[index]=expf(inp[index]-max_val)*norm;
        }
    }
}


baseline实验结果
使用shared memory

使用baseline的实现，我们只是 利用到了GPU并行的特性，但是GPU上的缓存(shared memory)我们是一点没有用到，所以这里我们来使用一下shared memory 来提升我们的运行速度。

首先，我们将数据传输给shared memory并让每个线程都进行一点点操作 ，减少线程空闲的现象

    extern __shared__ float shared[];
    float max_val = -INFINITY;
    for (int j = tid; j < C; j+= block_size) {
        max_val = fmaxf(max_val, inp[i+j]);
    }
    shared[tid] = max_val;// 就是这个块内 ，每个数据在block_size大小范围内的最大值，比如我现在block_size 是128 也就是一个block 块处理128个数据，那么shared[tid]仅代表自己，,如何block_size 是32 这种就表示可以128/block_size的数据的最大值
    __syncthreads();// 进行所有线程出的同步，防止现数据错误


利用 shared memory 完成max的操作

    for(int stride = block_size>>1; stride>0;stride>>=1){
        if(tid<stride){
            shared[tid] = fmax(shared[tid], shared[tid+stride]);
        }
    }
    __syncthreads();// 可能对应的tid也需要被其他的线程需要，所以需要同步更新一下
    // // 这个时候规约到了shared[0]
    float shared_max = shared[0];


接下来，需要更新将max操作得到的结果存回global memory并完成sum运算

    for (int j = tid; j < C; j += block_size) {
        out[idx * C + j] = expf(x[i+j] - shared_max);
    }
    __syncthreads();
  
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int j = tid; j < C; j += block_size) {
        sumval += x[j];
    }
    shared[tid] = sumval;//将输入值从全局内存 搬运到共享内存
    __syncthreads();
    // 规约求和
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();


最后就是进行softmax的答案求解

    float sum = shared[0];
    // divide the input values by the sum
    for (int j = tid; j < C; j += block_size) {
        out[idx * C + j] = x[j] / sum;
    }


全部代码如下

__global__  void softmax_forward_smem_gpu(float* out, const float* inp, int N, int C){
    // 每个线程负责一个元素的计算，也就是取代外层N的循环，在N维上并行起来
    extern __shared__ float shared[];
    int idx = blockIdx.x; // 位于第几块 
    int tid = threadIdx.x; // 块内第几个线程
    int block_size = blockDim.x;// 一个块多大
    int i =  block_size * blockIdx.x; // 线程这组数据的线程块位置
    // 计算每个线程块计算大小内的最大值->这里是每一个block计算一行
    const float* x = inp + idx * C; // idx-th row of inp
    float max_val = -INFINITY;
    for (int j = tid; j < C; j+= block_size) {
        max_val = fmaxf(max_val, inp[i+j]);
    }
    shared[tid] = max_val;// 就是这个块内 ，每个数据在block_size大小范围内的最大值，比如我现在block_size 是128 也就是一个block 块处理128个数据，那么shared[tid]仅代表自己，,如何block_size 是32 这种就表示可以128/block_size的数据的最大值
    __syncthreads();// 进行所有线程出的同步，防止现数据错误
    // 开始规约操作，计算最大值
    // 计算一个block 里面的最大值
    for(int stride = block_size>>1; stride>0;stride>>=1){
        if(tid<stride){
            shared[tid] = fmax(shared[tid], shared[tid+stride]);
        }
    }
    __syncthreads();// 可能对应的tid也需要被其他的线程需要，所以需要同步更新一下
    // // 这个时候规约到了shared[0]
    float shared_max = shared[0];
    // 用全局变量更新一下输入值
    for (int j = tid; j < C; j += block_size) {
        out[idx * C + j] = expf(x[i+j] - shared_max);
    }
    __syncthreads();
    // thread coarsening again, for the sum
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int j = tid; j < C; j += block_size) {
        sumval += x[j];
    }
    shared[tid] = sumval;//将输入值从全局内存 搬运到共享内存
    __syncthreads();
    // 规约求和
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int j = tid; j < C; j += block_size) {
        out[idx * C + j] = x[j] / sum;
    }

}

实验结果：

使用shared memory实验结果
擅用warp结构

线程束(warp) 是 SM 中基本的执行单元。一个线程束由32个连续线程组成，这些线程按照单指令多线程(SIMT)方式执行（即所有线程执行相同指令，每个线程在私有数据上操作）。

先进行线程块级别的操作，之后我们将进行warp级别的max操作，以提高执行效率

__device__ float warp_reduce_max(float val){
// warp 级别的最大值
    for(int offset = 32>>1;offset>0;offset>>=1){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFF,val,offset));// 与offset之后的数据进行操作
    }
    return val;

}
__global__  void softmax_forward_warp_gpu(float* out, const float* inp, int N, int C){
    .........
    float max_val = INT_MIN;
    for (int j = tid; j < C; j += block_size) {
        max_val = fmaxf(max_val, inp[i+j]);
    }
    max_val = warp_reduce_max(max_val);
    float offset = __shfl_sync(0xFFFFFFFF, max_val, 0);// 广播到warp的全部位置
}


全局代码如下：

__device__ float warp_reduce_max(float val){
// warp 级别的最大值
    for(int offset = 32>>1;offset>0;offset>>=1){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFF,val,offset));// 与offset之后的数据进行操作
    }
    return val;

}
__device__ float warp_reduce_sum(float val){
//warp级别的求和
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;

}
__global__  void softmax_forward_warp_gpu(float* out, const float* inp, int N, int C){

    // 问题：1、一个block里的thread应该多余32，而不是恰好是一个warp，应该先将数据处理完，然后最后进行warp的处理
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int i =  128 * blockIdx.x; // 线程这组数据的起始位置--> 为了验证warp的效果，外层的<<<128,32>>>一共有128个线程块，每个线程块32个线程，每个线程要覆盖4个数据，但是每一个线程块是计算一行的位置嘛，所以需要✖️128 而不是block_size，或者说是叫grid_dim.x
    // 计算每个线程块计算大小内的最大值->这里是每一个block计算一行
    float max_val = INT_MIN;
    for (int j = tid; j < C; j += block_size) {
        max_val = fmaxf(max_val, inp[i+j]);
    }
    max_val = warp_reduce_max(max_val);
    float offset = __shfl_sync(0xFFFFFFFF, max_val, 0);// 广播到warp的全部位置
    // 计算 sum求和
    for (int j = tid; j < C; j += blockDim.x) {
        out[idx * C + j] = expf(inp[i+j] - offset);
    }
    float sum_val = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        sum_val += out[i+j];
    }
    sum_val = warp_reduce_sum(sum_val);
    float sum = __shfl_sync(0xFFFFFFFF, sum_val, 0);
    float norm = 1.f / (float)sum;
    for (int j = tid; j < C; j += block_size) {
        out[i+j]=out[i+j]*norm;
    }
}

使用warp结构实验结果
使用shared memory + warp

上两节，介绍了warp优化和shared memory操作的优化 ，这节我们将他们结合起来

__global__ void softmax_forward_smem_warp_gpu(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warp_reduce_max(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum
    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warp_reduce_sum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

warp + smem的优化结果
优化online softmax

前文介绍了，safe softmax 的实现，其实online softmax和前文的优化手段是一样的，利用shared mem、减少线程的空闲，解决bank conflict、warp级别优化。所以 我们直接给出代码

__global__ void softmax_forward_online_smem_gpu(float* out, const float* inp, int N, int C){
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32; // 一共多少个warp

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    maxval = warp_reduce_max(maxval);
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    sumval = warp_reduce_sum(sumval);
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                y[i + u*blockDim.x] = reg_array[u] / sum;
            }
        }
    }

}

online优化后的版本
online baseline结果
参考

【手撕online softmax】

[Attention优化][2w字] 原理&图解: 从Online-Softmax到FlashAttention V1/V2/V3

ops(2)：SoftMax算子的 CUDA 实现
