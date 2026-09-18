# OpenCL Reduction优化

**作者**: 2know​东北大学 信号与信息处理硕士

**原文链接**: https://zhuanlan.zhihu.com/p/264786866

---

OpenCL-Reduction

本文通过OpenCL求和的例子，介绍OpenCL的reduction优化技巧。

Local Memory 基础版

在GPU上进行数组求和，最常用的方法是由多个work group分别对数组的不同部分求和，而后由cpu讲多个work group的结果进行加和。当数组十分巨大的时候，可以通过GPU进行两次求和，保证最后cpu计算部分数量很少。

如果在一些应用中为了降低数据从CPU到GPU的拷贝，也可以考虑最后由一个单独kernel进行求和。最后步骤数据量较少，开销不是很大。本位主要探讨第一步多个work group并行求和的过程。如下图所示：

假设数据长度为N，申请N个线程，

由N个线程负责从global memory中把数据加载到local memory中
偶数线程负责将自己（i）位置的数据与下一位置（i+1）位置的数据相加
被4整除的线程将自己位置（i）与自己下一位置(i + 2)位置的数据相加
被8整除的线程将自己位置（i）与自己下一位置（i + 4）位置的数据相加
如此循环直到局部内存中的数据全部被累加到local memory的0号位置
各个work group的索引为0的线程把数据写入global memory

以上是最基本的实现方案，核心代码如下：

    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
​
    for(unsigned int s=1; s < get_local_size(0); s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

方案总结：

优点

实现了数据加和的并行
work group内部累加的时候使用local memory 提升了访存性能




缺点

每次都使用进入for循环的线程数量的一半， 缺少完整warp运行，即存在线程分支
取余操作效率低
local memory的访问可能存在banck conflict




基于以上分析， 我们对该方案进行第二轮优化。

Local Memory 优化版

仔细想一想，上一版的数据加载流程虽然使用了local memory， 但是从global加载数据的时候，是存在可优化空间的；例如，我们只使用一半的线程，第一次加载一半数据，第二次加载另一半的时候直接累加到第一次加载的数据上。这样做可以减少N/2次的local memory读写，同时local memory的使用只有一半。

关于取余操作和每次运行warp的一半线程，本质上是因为使用偶数线程进行计算导致的，如果每次计算只用workgroup中的前一半线程，那么可以避免取余操作，同时在数据较多的前n次for循环中，warp是完成的。

可能出现banck conflict的原因是每个线程随着循环次数的增加，线程访问数据步长越来越大很有可能会引起banck conflict。如果每次使用连续的线程操作，那么，线程间的步长是固定的，通过调整workgroup的size可以有效避免banck conflict。

具体操作如下图：

按照如图所示的改进方案修改后的核心代码如下：

    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    if (i + get_local_size(0) < n) 
        sdata[tid] += g_idata[i+get_local_size(0)];  
​
    barrier(CLK_LOCAL_MEM_FENCE);
​
    // do reduction in shared mem
    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

注意这次申请的线程数量是基础版的一半

本方案解决可基础版的缺点，性能会有很好的提升，对于一般应用，这样的性能可能已经够用；但是代码依然还有优化空间。从源代码看，目前单线程在第一次加载的时候直接进行一次加法节省了一次从local memory的读写操作；但其实按照这个思路，可以节省更多次的local memory读写，那么多少次最合适呢？所以这是一个待优化的点。另外看，累加部分，for循环中每次计算都是需要同步的，熟悉GPU的同学应该知道，同步的代价是比较高的，减少同步，可以有效提升程序性能。所以接下来我们就从这两个点切入做进一步的优化。

优化掉不必要的同步

既然要优化掉不必要的同步，那么就需要思考一个问题，同步的目的是什么，为什么要进行同步。仔细思考不难想到，同步是因为a需要用b的结果；例如上一节for循环中的reduce，每次循环都需要同步，便是因为下一次循环需要使用上次循环的结果，因此需要同步，确保计算结果被写入内存。这么思考似乎每次同步都是必要的。

再思考GPU的运行机制，GPU严格意义上的并行是warp内部的并行，warp之间并不是完全的真正意义的并行，是由先后调度的。所以warp之间的顺序是无法保证的，因此为了确保数据一致性，我们需要做同步；但是warp内部，是严格意义上的并行，多个线程执行的是同一条指令，并且是同步完成的。换句话说，warp内部天然是同步的。不存在哪个线程先执行，哪个线程后执行。

到这里，思路就比较明确了，我们要利用warp内部线程天然同步的特性，来减少不必要的同步；具体优化代码如下：

    // do reduction in shared mem
    #pragma unroll 1
    for(unsigned int s=get_local_size(0)/2; s>32; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
​
    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }
​

代码思路比较清晰，我们把最后一个运行的warp从for循环中提取出来展开；这样避免了最后一个warp的同步。这里有两个变化：

for循环的终止条件由s>0 变为s > 32
tid < 32的case下，没有屏蔽掉多余的线程

首先说明一下blockSize，这个参数是一个常数，一般取2的次幂。假定blockSize=256；

32是基于GPU的每个warp的thread数量来确定的，该参数可以通过具体info参数获取，不同架构GPU会有变化。目前桌面端大多数GPUwarp包含32个线程，因此这里取32.关于warp大小的优势和劣势，这是GPU硬件设计需要考量的事情，会在后续文章中进行讲解，这里不展开，只需要知道该参数对于具体GPU是一个固定数据。

对于展开的if部分，初学者可能会有疑惑，因为想象中的代码应该是如下所示：

 if (tid < 32)
    {
        if (blockSize >=  64 && tid < 32) { sdata[tid] += sdata[tid + 32]; }
        if (blockSize >=  32 && tid < 16) { sdata[tid] += sdata[tid + 16]; }
        if (blockSize >=  16 && tid <  8) { sdata[tid] += sdata[tid +  8]; }
        if (blockSize >=   8 && tid <  4) { sdata[tid] += sdata[tid +  4]; }
        if (blockSize >=   4 && tid <  2) { sdata[tid] += sdata[tid +  2]; }
        if (blockSize >=   2 && tid <  1) { sdata[tid] += sdata[tid +  1]; }
    }

两段代码的区别在于是否屏蔽掉多余线程，屏蔽掉线程的代码，容易理解，但是显然是严重影响性能的；因为一个warp内部的线程存在分支，会严重影响warp的执行效率。

那么，不屏蔽warp内部的多余线程，计算结果是否正确呢？

第一个if容易理解，0号线程完成0号位置加32号位置，一次类推，32个线程都最一组加法；

但是第二个if的时候，目前总共有32个线程，0号线程完成0号位置加16号位置，那么有一个问题，16号线程会做16号位置和32号位置的加法；那么0号线程取的16号位置是16号线程加之前还是加之后的值呢？如果是加之后，那结果不就错了吗？

我们先看warp执行的指令，warp需要执行6组加法指令，每一组加法指令包含如下三条指令(示意并非真正指令)：

x = load(tid, sdata);
y = load(tid + m, sdata);
x += y;

根据warp内所有线程是执行相同指令处理不同数据的特性；所有线程都会首先，加载自己对应位置数据，此时，0-32号线程数据加载正确，下一条指令，所有线程加载步长16的位置的数据，0-32号线程加载16-48位置数据，目前x和y的值都正确；最后一步，所有线程把自己持有的两个数据加和更新到自己对应位置，即x+y被更新0-32；这时候考量各个线程的计算结果，0-15是正确的数据，16-31线程数据本身是无效数据，不予考虑。

可以看出warp内部的同步执行，保证了0号线程使用的16号位置的值，是16号线程更新之前的值，所以结果是正确的。到此我们完成了对最后一个warp的同步优化，有效避免了最后一个warp的同步，也避免了最后一个warp内部的线程分支。

以上是对线程同步及线程分支的优化。




for循环展开

完成对分支指令及同步指令的优化后，代码如下：

  // do reduction in shared mem
    for(unsigned int s=get_local_size(0)/2; s>32; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
​
    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }

观察代码不难发现，for循环是可以展开的，我们知道，对于确定架构的GPU，work group的size是由最大值的，一般为1024,512,256居多；这里姑且设为1024；那么，blockSize的最大值也就是1024，根据我们的计算方法，每个线程每次完成步长为s的两个数的加法，为了利用local memory，并且满足线程访存合并等前文优化的条件，我们s的最大值为blockSize/2;所以数据固定后，我们是可以对for循环进行手动展开的，展开如下：

 // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    
    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }

for循环展开与CPU上的for循环展开是相同的原理，可以有效提高指令的并行性，属于指令级优化的常用方法。

到此为止，整个程序已经全部完成优化，所有指令都做到最简，几乎没有冗余，那么程序是否还有优化空间呢？优化其实就像海绵中的水，只要你愿意挤，总还是能挤出来点，只是越挤越少罢了。

回到前文提的一个点，线程在加载数据的时候，从global中把数据加载到local中，前文做了一个优化，如果先加载一半数据，剩下的一半加载后直接加到前一半数据上，可以有效降低local的消耗和数据加载次数；沿着这个思路，如果先加载4分之一呢？

单线程计算数量优化

按照前文的思路，我们知道，单线程计算的数据越多，local消耗越少，local中数据的读写次数也越少；但是由于单线程计算数量增多，线程数量变少，极限情况是单线程串行计算，显然这样并不是最优的；那么多少合适呢？

优化代码如下：

 unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);
    unsigned int gridSize = blockSize*2*get_num_groups(0);
    sdata[tid] = 0;
​
    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        sdata[tid] += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            sdata[tid] += g_idata[i+blockSize];  
        i += gridSize;
    } 
​
    barrier(CLK_LOCAL_MEM_FENCE);
​
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    
    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];

在不同的架构的GPU上用户通过调整不同的blockSize，寻找最优的的blockSize值，来达到最优性能。

总结

到此为止，我们完成了，对ocl中reduction的优化介绍，以这样一个例子，向大家介绍了GPU程序优化的基本思路：

数据并行
使用local memory提升程序的访存性能
避免banck conflict
访存合并
避免warp内部存在过多分支指令
避免不必要同步
适度的for循环展开
合适的block划分

以上属于GPU程序优化中的基础手段，也是最有效的手段，对于大多数算法来说，从cpu移植到GPU上通过以上方法，就可以获得不错的性能提升；在后两种优化中，大家也能看出，是结合算法的特点及硬件的特性，进行优化，有些参数也需要根据具体硬件测试确定；所以，更进一步的优化及性能提升，也是是需要结合具体算法进行针对性优化，后续我们会针对具体算法介绍更多特定算法特有的优化方法。希望以上介绍的通用优化方法，能够帮助大家快速实现已有CPU程序到GPU的优化移植，并获得很好的性能。
