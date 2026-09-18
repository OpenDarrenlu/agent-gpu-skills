# 浅谈cuda graph在llm推理中的应用

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/715863693

---

​
目录
收起
参考
一个简单的demo
cuda graph 在 vllm 中的使用
关于控制流节点的一点个人理解
1. 场景一，condition 不依赖于上一个算子的执行结果
2. 场景二，condition 依赖于上一个算子的执行结果
结语
附录
参考

快速过下这篇 blog，了解 cudagraph 的作用和基本原理：改名机会要过期：CUDA效率优化之：CUDA Graph

这是翻译自 https://developer.nvidia.com/blog/cuda-graphs/ 的 blog，也可以阅读原篇。

这篇 游凯超：一文读懂cudagraph 更深入些，好像是 nv 官方的大佬写的。

上面是基于 C++ api 的讲解，关于在 torch 中如何使用 cuda graph 可以阅读这篇 blog（blog有点老了）：

Accelerating PyTorch with CUDA Graphs

本 blog 主要学习下torch cuda graph 在 LLM （以vllm为例）推理中的应用。

一个简单的demo

cuda graph 的作用就是可以减少 kernel launch 的开销。在某些场景下，如有大量的 kernel 的实际运行时间很短，甚至超过了 kernel launch 的时间，这时候有一定的性能收益。

上述官方在 torch 中使用 cuda graph 的文档有点老了且包含了训练过程，这里提供一个更简单明了的demo。长话短说，直接上代码吧，假设有一个自定义的模型，代码如下:

class simpel_model(nn.Module):
    def __init__(self):
        super().__init__()
        num_layer = 10000
        self.blocks =  torch.nn.ModuleList([nn.Linear(D_in, D_out) for _ in range (num_layer)])
    
    def forward(self, x, y, z):
        a = torch.matmul(x, y)
        b = torch.matmul(x, z)
        c = torch.add(a, b)
        for block in self.blocks:
            c = block(c)
        return c

这个模型没有任何实际意义，可以将 Liner 的 D_in, D_out, 设置得小一点（如32），模拟 kernel 实际执行时间很短得场景，当然你可以定义更有实际意义的model，如 llama2 这种。这个模型定义了三个输入，计算主要由 Linear 组成。

接下来定义一个 CUDAGraphRunner 类：

class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = {}
        self.graph_output = {}
    
    def capture(self, x, y, z):
        assert self.cuda_graph is None

        self.cuda_graph = torch.cuda.CUDAGraph()
        # self.cuda_graph.enable_debug_mode()
        with torch.cuda.graph(self.cuda_graph):
            out = self.model(x,y,z)
        torch.cuda.synchronize()
        # self.cuda_graph.debug_dump("graph.dot")

        # 定义 graph 输入 placeholder
        self.graph_input['x'] = x 
        self.graph_input['y'] = y
        self.graph_input['z'] = z
        # 定义 graph 输出 placeholder
        self.graph_output['output'] = out 
        
    def forward(self, x, y, z):
        self.graph_input['x'].copy_(x)
        self.graph_input['y'].copy_(y)
        self.graph_input['z'].copy_(z)
        self.cuda_graph.replay()
        return self.graph_output['output']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

然后就可以使用这个 CDUAGraphRunner 的 wrapper 对模型进行 cuda graph 加速了：

model = simpel_model().cuda()
inp = torch.randn(32, D_in).cuda()
model.eval()
model(x=inp, y=inp, z=inp) # warm up, 触发一些 gpu 资源的初始化

graph_runner = CUDAGraphRunner(model)
inputs = {"x":inp, "y":inp, "z":inp}
graph_runner.capture(**inputs)
graph_runner(**inputs) # cuda_graph_runner run

model 原生推理和cudagraph 推理性能对比完整的测试代码见最后附录。

性能对比如下（4090的机器上）：

cuda_graph_elasped_time: 43.5736572265625 ms, ori_infernce_elasped_time: 185.53947265625 ms

可以使用 nsys profile 下看看：

cuda graph inference




model origin inference

需要注意的几点是：

在 capture 之前，一般让模型先 warm up 跑一次，主要是触发一些库资源的初始化，如 cublas handle。
capture 过程中，模型不会真正的执行（算子不会真正的执行），而是在生成一个静态图，计算图可以理解为是由算子和变量两种节点组成的。
在 graph 执行时（forward函数中），需要将输入拷贝到计算图的输入变量节点中。
计算图不支持动态shape

下面是一个根据该模型的组网代码画的对应的计算图的示意图（方框表示变量节点，圆圈表示计算节点（俗称op））：

cuda graph

实际的计算图，可以通过

self.cuda_graph.enable_debug_mode()
self.cuda_graph.debug_dump("graph.dot")

打印，打开 dot 图可以看出，实际计算图就是一串连续的算子（可以认为是上述示意图的拓扑排序以及将while循环展开），这里就不放图了，读者可以自行运行代码生成 dot 文件并打开。

cuda graph 有独立的计算空间（内存pool）,所以需要将外部的 torch tensor copy 到计算图的输入变量节点中。此外由于计算图不支持动态shape，所有如果实际计算时的shape与 capture 时的输入的shape不一致则会报错，如下：

model = simpel_model().cuda()
inp = torch.randn(32, D_in).cuda()
model.eval()
model(x=inp, y=inp, z=inp) # warm up, 触发一些 gpu 资源的初始化，包括一些中间 tensor 的创建

graph_runner = CUDAGraphRunner(model)
inputs = {"x":inp, "y":inp, "z":inp}
graph_runner.capture(**inputs) # cpature

input = torch.randn(64, D_in).cuda()
inputs = {"x":input, "y":input, "z":input}
graph_runner(**inputs) # cuda_graph_runner run

上述 demo 中，构建 cuda graph 时，输入的 shape 的第一个维度为 32，实际运行时的 shape第一个维度为 64，将这种 shape 的 tensor 拷贝到图的输入变量节点时，就会报错，如下：

RuntimeError: The size of tensor a (32) must match the size of tensor b (64) at non-singleton dimension 0
cuda graph 在 vllm 中的使用

vllm 中关于 cuda gprah 的应用在以下文件中：

https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py#L1503

主要是 ModelRunner 和 CUDAGraphRunner 两个类。

核心逻辑是：

将模型封装为多个 CUDAGraphRunner，不同的 batch 对应不同的 CUDAGraphRunner，一个 batch 对应一个 CUDAGraphRunner，在 ModelRunner 执行模型的时，根据输入的 batch 不同，寻找匹配的 CUDAGraphRunner，如果找不到，则回退直接调用 model.forward。

LLM 模型中，prefill 阶段输入的 batch，seq_len， 两个维度不可控，因此只针对 generate 过程使用 cuda_graph，提前设置一批 batch，针对每个不同 batch capture 住一个 cudagraph，运行时根据输入的 shape 找到匹配的 cuda_graph_runner 即可。

我将上述逻辑抽象出来，写成一个简单的 demo 如下：

import torch
import torch.nn as nn

D_in = 1024
D_out = 2048
class ModelRunner():
    def __init__(self, model):
        self.model = model
        self.graph_runners = {}  # (int, CUDAGraphRunner)

    @torch.inference_mode()
    def capture_model(self):
        for batch in [1, 2, 3, 4]: # 提前设置一批 batch
            input = torch.randn(batch, D_in).cuda()
            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(input)
            self.graph_runners[batch] = graph_runner
    
    @torch.inference_mode()
    def execute_model(self, x):
        batch = x.size(0)
        if batch in self.graph_runners:
            model_executable = self.graph_runners[batch] # 根据输入找到对应的 graph_runner
        else:
            print(f"warning, no cudagraph_runner, back to origin model")
            model_executable = self.model # 回退到原始的 model
        return model_executable(x)


class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = None
        self.graph_output = None
    
    def capture(self, x):
        assert self.cuda_graph is None

        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            out = self.model(x)
        torch.cuda.synchronize()

        self.graph_input = x # 定义 graph 输入 placeholder
        self.graph_output = out # 定义 graph 输出
        
    def forward(self, x):
        self.graph_input.copy_(x)
        self.cuda_graph.replay()
        return self.graph_output
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# 创建模型和输入数据
model = nn.Linear(D_in, D_out).cuda()
model.eval()
input = torch.randn(4, D_in).cuda()
output_ref = model(input)

model_runner = ModelRunner(model)
model_runner.capture_model() # model_runner 构造cuda graph
output = model_runner.execute_model(input) # 执行

torch.testing.assert_close(output_ref, output, rtol=1e-03, atol=1e-03)

在 vllm 中，设置的 capture 的batch 为：

设置得越多，构建 cudagraph 时耗费的显存资源也越多（保存的计算图更多）。

此外，如果 decode 时，batch是3，则 pad 到4，就可以匹配到对应的 cudagraph runner了。

关于控制流节点的一点个人理解

首先明确一点，cuda graph 支持控制流节点。

https://developer.nvidia.com/zh-cn/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/

但看起来是需要通过手动组网的方式，即构建graph之后，然后插入一个控制流节点，设置控制流节点的 condition 以及执行主体（body）等。

但在 vllm 以及多数基于 pytorch 组网的模型中，是通过下面这种方式使用 cuda graph 的：

self.cuda_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(self.cuda_graph):
     out = self.model(x)

这看起来像是一个黑盒，个人的理解这段代码就是动转静的功能。

1. 场景一，condition 不依赖于上一个算子的执行结果
import torch
import torch.nn as nn

D_in = 32
D_out = 32
torch.manual_seed(1)

class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = {}
        self.graph_output = {}
    
    def capture(self, x, condition):
        assert self.cuda_graph is None

        self.cuda_graph = torch.cuda.CUDAGraph()
        self.cuda_graph.enable_debug_mode()
        with torch.cuda.graph(self.cuda_graph):
            out = self.model(x, condition)
        torch.cuda.synchronize()
        self.cuda_graph.debug_dump("graph.dot")

        # 定义 graph 输入 placeholder
        self.graph_input['x'] = x 
        # 定义 graph 输出 placeholder
        self.graph_output['output'] = out 
        
    def forward(self, x, condition):
        self.graph_input['x'].copy_(x)
        self.cuda_graph.replay()
        return self.graph_output['output']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# 创建模型和输入数据
class simpel_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj =  nn.Linear(D_in, D_out)
    
    def forward(self, x, condition):
        if condition:
            out = torch.add(x, 1)
        else:
            out = self.proj(x)
        return out

model = simpel_model().cuda()
model.eval()

inp = torch.randn(32, D_in).cuda()
model(inp, condition=False) # warm up, 触发一些 gpu 资源的初始化

graph_runner = CUDAGraphRunner(model)
graph_runner.capture(inp, condition=False)

output = graph_runner(inp, condition=True) # cuda_graph_runner run
output_ref = model(inp, condition=True)

torch.cuda.synchronize()
torch.testing.assert_close(output_ref, output, rtol=1e-03, atol=1e-03)

该 demo 中，capture 时，并未完全捕捉到条件控制节点的两条路径，而是只能捕捉到实际运行的算子路径（类似torch.jit.trace)，只能 trace 被激活的条件分支，所以 demo 中，cudagraph trace 到了 condition 为false 时的那条分支，但是实际运行时走 condition 为 true 的那条分支，这样就会导致结果出错。

幸运的是，在 llm model 的组网中，虽然有一些 if 等控制流， 如 if bias 语句等，但是 condition 都是模型的 config 文件中确定的数值，所以 llm 模型推理中的算子路径是确定的。

2. 场景二，condition 依赖于上一个算子的执行结果

这种场景复杂些，在cuda graph 动转静时会失败，原因是，上一个 gpu 算子的结果是在 gpu 上，在python cpu 代码层面判断上一个算子的结果，然后根据结果的不同去调不同的算子，有隐藏的内存拷贝和同步逻辑。cuda graph 中，不支持同步逻辑。（这种控制流算子相当于一个 CPU 算子），如下面这种组网：

class simpel_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj =  nn.Linear(D_in, D_out)
    
    def forward(self, x):
        if x.sum() > 0:
            out = torch.add(x, 1)
        else:
            out = self.proj(x)
        return out

这里相当于把上一个算子（sum运算）的执行结果拷贝到 cpu 上，python cpu 端进行判断，然后调用不同的算子，由于拷贝时，隐藏同步逻辑，会导致 cuda graph 构建报错。

隐藏的内存拷贝和同步逻辑

最开始说的 cuda graph 支持控制流算子，相当于将这个 ”CPU控制流算子“变成 ”GPU控制流算子“，这样不需要拷贝操作了，需要一些 api 手动在 cuda graph 插入这种控制流算子。

或者使用 torch compile 可以避免这种问题。代码如下：

model = simpel_model().cuda()
model.eval()
model_opt = torch.compile(model, mode='reduce-overhead')
inp = torch.randn(32, D_in).cuda()
output = model_opt(inp) # warm up

with nvtx.annotate("model run", color="red"):
    output = model_opt(inp)
    torch.cuda.synchronize()
torch compile 中会构建cuda graph
结语

vllm 推理中 cuda graph 主要用于 decode 阶段的加速，大部分模型实际有益有限（一般5%左右的收益）。因为本身 cpu 上的kernel launch 和 gpu 上的实际 kernel 运行是一个异步过程（在launch当前kernel时，gpu也没闲着，在跑上一个kernel），同时绝大部分 kernel 的实际运行时间是远大于 kernel lauch 的启动时间的。

可以根据不同的实际业务场景，合理设置 vllm 中的 capture cuda graph 时的batch size，避免太多导致显存不足，同时提高 cuda graph runner 的命中率。

附录

cuda graph 优化前后的性能对比测试代码：

import torch
import torch.nn as nn
import os
import nvtx
import time
import gc
from torch.profiler import profile, record_function, ProfilerActivity

D_in = 32
D_out = 32
torch.manual_seed(1)

class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = {}
        self.graph_output = {}
    
    def capture(self, x, y, z):
        assert self.cuda_graph is None

        self.cuda_graph = torch.cuda.CUDAGraph()
        self.cuda_graph.enable_debug_mode()
        with torch.cuda.graph(self.cuda_graph):
            out = self.model(x,y,z)
        torch.cuda.synchronize()
        self.cuda_graph.debug_dump("graph.dot")

        # 定义 graph 输入 placeholder
        self.graph_input['x'] = x 
        self.graph_input['y'] = y
        self.graph_input['z'] = z
        # 定义 graph 输出 placeholder
        self.graph_output['output'] = out 
        
    def forward(self, x, y, z):
        self.graph_input['x'].copy_(x)
        self.graph_input['y'].copy_(y)
        self.graph_input['z'].copy_(z)
        self.cuda_graph.replay()
        return self.graph_output['output']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# 创建模型和输入数据
class simpel_model(nn.Module):
    def __init__(self):
        super().__init__()
        num_layer = 10000
        self.blocks =  torch.nn.ModuleList([nn.Linear(D_in, D_out) for _ in range (num_layer)])
    
    def forward(self, x, y, z):
        a = torch.matmul(x, y)
        b = torch.matmul(x, z)
        c = torch.add(a, b)
        for block in self.blocks:
            c = block(c)
        return c


def timed(fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    repeat = 10
    start.record()
    for _ in range(repeat):
        result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / repeat 

model = simpel_model().cuda()
inp = torch.randn(32, D_in).cuda()
model.eval()
model(x=inp, y=inp, z=inp) # warm up, 触发一些 gpu 资源的初始化
graph_runner = CUDAGraphRunner(model)
inputs = {"x":inp, "y":inp, "z":inp}
graph_runner.capture(**inputs)
graph_runner(**inputs) # cuda_graph_runner warm up

input = torch.randn(32, D_in).cuda()
output, cuda_graph_elasped_time = timed(graph_runner, **inputs)
output_ref, ori_infernce_elasped_time = timed(model.forward, **inputs)

torch.cuda.synchronize()
torch.testing.assert_close(output_ref, output, rtol=1e-03, atol=1e-03)
print(f"cuda_graph_elasped_time: {cuda_graph_elasped_time} ms, ori_infernce_elasped_time: {ori_infernce_elasped_time} ms")
