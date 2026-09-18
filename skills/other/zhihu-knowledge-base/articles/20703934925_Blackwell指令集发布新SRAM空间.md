# Blackwell指令集发布新SRAM空间

**作者**: CPAPCFGPA4.0 | Political Philosophy and Pre-Law | Music Production and Engineering

**原文链接**: https://zhuanlan.zhihu.com/p/20703934925

---

这几天查PTX文档时突然看到多了一栏：TensorCore 5th Generation Family Instructions，要求sm100以上，就是Blackwell新的变化了。

前缀为tcgen05，独立于wmma、wgmma指令空间。

更重要变化是新增可动态分配的Tensor Memory，定义为片上内存专供Tensor Core使用，既然是片上内存就当是SRAM了。

Tensor Memory在每个CTA被分为四块chunks对应每个warp私有访问。回想一个SM有四个quadrants，每个quad有一个独立warp dispatcher私有一组ALU和RF，Tensor Memory也跟着quad设计和布线的。

tcgen05支持从shared memory和RF读取写入数据，寄存器的ld/st和shared memory的cp都是异步指令。异步cp同时支持在线浮点提精，目前给了例子从fp4->fp8 fp6->fp8的在线promote。有点用，但不多，fp4和fp8这么一点精度，用处都比较局限。

swizzle最大边界还是128B swizzle。

在tcgen05 ISA内，指令序和内存序进一步弱化，引入两条手动fence指令声明同thread内的前后序

tcgen05.fence::before_thread_sync ;
tcgen05.fence::after_thread_sync  ;

意思是在tcgen05 ISA内，只要遵守fence指明的顺序，硬件可以重排指令乱序执行。tcgen05 ISA独立的内存一致性模型，单独指出：只有pipeline指令和同步指令执行序跟程序序一致，所有tcgen05指令默认没有任何执行序保证。整的跟DEC Alpha一样超弱的内存模型 Memory Model

Pipeline指令指tcgen05指令中带有.cta_group::N修饰符且N相同的前后指令，执行序一定符合程序序。

至此，Tensor Core已经变成一个很完整的实体：它是个外挂在SM quadrant的独立协处理器，他有独立的内存独立的ISA独立的内存序执行序，跟它通信还靠异步IO。以后nv卡可能会成为架构历史/教学上很有趣的一笔，他的SM quadrant是顺序发射顺序执行强内存序，但外挂一个Tensor Core乱序执行超弱内存序，真是神奇混搭。

就在4天前，ThunderKittens已经支持tcgen05指令集。据我所知，黄皮衣于1月24日发布PTX 8.7标准，ThunderKittens几乎在两三天内就push了tcgen05支持的branch。担心cutlass/cute第一方何时支持tcgen05的不用操心了，直接用ThunderKittens即可。

PTX 8.7发布的Blackwell ISA specs分为好几版，有sm_100 sm_101(应该是Thor端侧片) sm_120(RTX消费片)。
