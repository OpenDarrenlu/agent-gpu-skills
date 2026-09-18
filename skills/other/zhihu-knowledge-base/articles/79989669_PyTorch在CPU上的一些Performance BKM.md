# PyTorch在CPU上的一些Performance BKM

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/79989669

---

这里简单介绍一下用PyTorch在CPU上的一些性能相关的BKM。

内容以inference为主，毕竟CPU上主要的场景还是inference；另外这里CPU都指的是Intel Xeon.

gist里面写了英文版的，内容和这里的基本相当： General guidelines for CPU performance on PyTorch

1. 使用ChannelsLast

目前在PyTorch里面在CPU上可以选择3种memory format，分别是：

torch.contiguous_format: 这个就是默认的memory format，一般叫做NCHW。
torch.channels_last: 一般叫做NHWC。
torch._mkldnn: 也就是mkldnn的blocked memory format。准确的说torch._mkldnn是做为layout而并不是memory format来处理的（为了创建新的TensorTypeId），但这个地方并不关键。

使用方法很简单：

### 1. default (NCHW)
output = model(input)

### 2. channels last
input = input.to(memory_format=torch.channels_last)
model = model.to(memory_format=torch.channels_last)

### 多数CV模型第一层都是Conv，在Conv里面NHWC的优先级要比NCHW高
###（input或者weight有一个是NHWC就会走NHWC的path）
### 然后channels last会在整个模型propagate，所以只转model就可以。

### 3a. mkldnn blocked format (inference)
### torch.utils.mkldnn.to_mkldnn会完成Conv与Linear weights的prepacking
input = input.to_mkldnn()
model = torch.utils.mkldnn.to_mkldnn(model)
output = model(input)

### 3b. mkldnn blocked format (training)
input = input.to_mkldnn()
output = model(input)

如果model里面有某个operator不支持NHWC的话会被当做non-contiguous的NCHW处理，本身并不会计算错，但会打破channels last传递的链条，也就是后续operator会走NCHW的path，也就是可能会变慢。

如果model里面有某个operator不支持mkldnn layout，那就麻烦一些，需要手动在forward里面插入to_mkldnn()和to_dense()，不然会报runtime error：

class MyModel(nn.Module):
    def __init__(self):
        self(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(10, 10, 3)
        # MyModel has mkldnn unsupported operators X()
        self.unsupported_mod = nn.X()
        self.linear1 = nn.Linear(10, 20)
        
    def forward(self, x):
        x = self.conv1(x)
        # use default layout for module without mkldnn support
        x = x.to_dense()
        x = self.unsupported_mod(x)
        x = x.to_mkldnn()
        x = self.linear1(x)
        return x

这个过程你可以想象成cuda的backend有个operator不支持，需要在cpu上面跑，处理方式是类似的。

更多关于channels last优化相关信息，可以查询PyTorch Channels Last Memory Format Performance Optimization on CPU Path

关于channels last性能对比，可以查询convnet-benchmark-py

Results on Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz, single socket with 20 cores available here.

### NCHW run
Running on torch: 1.8.1+cpu
Running on torchvision: 0.9.1+cpu
ModelType: resnet50, Kernels: nn Input shape: 1x3x224x224
nn                              :forward:      55.89 (ms)      17.89 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      55.89 (ms)      17.89 (imgs/s)

### NHWC run
Running on torch: 1.9.0a0+git850a6bd
Running on torchvision: 0.10.0a0+4f34ae5
ModelType: resnet50, Kernels: nn Input shape: 1x3x224x224
nn                              :forward:      14.02 (ms)      71.31 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      14.02 (ms)      71.31 (imgs/s)
2. TorchVision使用channels last

如果model里面使用了torchvision的csrc模块，例如"ROIAlign"，那么torchvision本身也需要有channels last的支持。

关于MaskedRCNN的优化工作记录在mingfeima/detectron2

性能对比：

Results on Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz, single socket with 20 cores:

### with config "fast_rcnn_R_50_FPN_1x.yaml"
### NCHW (torch-1.8.1/vision-0.9.1): 300 iters in 326.0195782049559 seconds.
### NCHW (torch-opt/vision-0.9.1): 300 iters in 185.4384527085349 seconds.
### NCHW (torch-opt/vision-opt): 300 iters in 80.56146793198423 seconds.
### NHWC (torch-opt/vision-opt): 300 iters in 55.49435344198719 seconds.

后续upstreaming及优化工作还在继续。

3. 环境变量设置

如果是single instance, 需要限制OpenMP thread数量以及CPU binding的方式，如下：

export OMP_NUM_THREADS=[number_of_physical_cores]
export KMP_AFFINITY=granularity=fine,compact,1,0

如果是dual socket的CPU而只想用single socket跑，为了避免remote memory access，需要限制 numactrl：

# e.g. say each socket has 20 cores, to use the 1st socket:
numactl --physcpubind=0-19 --membind=0

如果是multi instance，每个instance都会展开独立的OpenMP thread pool，也就是每个instance都需要限制 OMP_NUM_THREADS。保证 omp_threads * num_instances 不会超出物理核的数量，不然就会over subscription，就是抢核，这种情况对于Xeon来说是灾难性的。

multi instance 的情况要比single instance复杂很多，因为上层的threading model可能有很多种，可以是 torch.multiprocessing, std::threads, TBB等等。另外，设置affinity的方式要具体问题具体分析。社区上遇到很多这种multi instance没设置好环境变量的，问题五花八门，一般来讲分析此类问题最方便的就是用vtune，先查一查多少个omp master在同时跑。

4. 使用jemalloc

PyTorch使用的是动态图，本身有一个缺点，就是output每次都是重新分配内存的。batch size比较大的时候memory allocation的开销是很大的（clear page），batch size为1的时候开销很小。 这个问题通过 jemalloc 可以得到一定程度的缓解，jemalloc是替换libc中的malloc的。tcmalloc或者tbbmalloc功能和jemalloc类似。

如果在vtune中观察到clear_page占比很高，那就是要考虑使用jemalloc了。jemalloc本身有一个限制是不能跨numa node，一般跨numa node要考虑使用multi instance。

### jemalloc
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.1/lib/libjemalloc.so

### tcmalloc
export LD_PRELOAD=/home/mingfeim/packages/gperftools-2.8/install/lib/libtcmalloc.so
5. 用ICC编译PyTorch

PyTorch可以使用icpx编译，不过鉴于目前PyTorch explicit vectorization的方法，icc不会带来显著性能提升。

CC=icx CXX=icpx python setup.py build
6. DataLoader

torch.utils.data.DataLoader里面现在有个bug，我测试结果是num_workers > 0反而会变慢。实测的时候可以和num_workers = 0比较一下。TODO: 以后有时间处理掉。
