# 拒绝autotune，如何推算triton 2D kernel参数？

**作者**: 白牛github tpoisonooo

**原文链接**: https://zhuanlan.zhihu.com/p/2002050606311952981

---

昨天手糙 online-softmax（一篇nv的老论文），瞥见 triton tutorial有暴搜就很别扭。好的 MNK 应该是算出来的对吧？

糙了一个 launch 参数推导在 gayhub：

https://github.com/tpoisonooo/nano-vllm/blob/main/bench/utils.py
github.com/tpoisonooo/nano-vllm/blob/main/bench/utils.py

假设整个 kernel 是 2D 形式的，layernorm、softmax、rmsnorm 都行，这个函数会根据<长、宽、block变量数> 推导以下参数：

class LaunchParam:
    def __init__(self):
        self.block_size = 128
        self.num_warps = 1
        self.num_stages = 2
        self.rows_per_prog = 1

其中“block变量数”，是指一个 for _ in range() 中， 出现的tl.load 变量个数，用来计算每个 thread 占用的寄存器 。 例如这段代码里：

    cols = tl.arange(0, BLOCK_SIZE)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Out + cols, y, mask=mask)

w/b/x/y 都占着 BLOCK_SIZE 寄存器，所以“block变量数”是 4。

最终效果挺好：

persistent 粉色是 online-softmax + launch 参数
v2 是普通的 online-softmax
v1 是 naive 5次读数据的版本
tc 就是 torch.compile
torch.softmax
0x01 计算思路

代码里写了注释，主要在算 <block 个数, reg> 两个维度。

STEP1 先决定 BLOCK_SIZE，受限于 tensor.width。不超过 2048

STEP2 定住了 BLOCK_SIZE，同时kernel 复杂度是已知的（例如前文的 4），就能推算出每个 block 需要多少个 reg

STEP3 SM 个数、最大 regfile 都是固定的物理参数，那就知道这个 kernel 想打满 GPU 需要跑多少个 block

自然就知道 persistant 模式中每个 program 要算多少行数据：

        total_blocks_needed = NUM_SM * max_blocks_per_sm * target_occupancy
        self.rows_per_prog = max(1, math.ceil(n_rows / total_blocks_needed))
0x02 闲谈 auto.tune 有啥缺陷

tune 本身没问题——ncnn/megengine/tvm也都有——有缺陷的是提前不tune、延后到运行时才tune。

导致曾经修了无数个精度对不齐的 jira。

同样的物理机、同样的软件版本：

早上 GPU 渲染开多了，面部识别用 im2col conv 是一个 feature
晚上走 wino conv 又是另一个 feature

这不要命么?

那我提前 tune 行不行？

好，每次发版本。P4/T4/3090/V100/A100/H20/Bxx 各打一份配置吧。

想想也挺要命，总感觉是把矛盾从 infra 转移给了产品线。
