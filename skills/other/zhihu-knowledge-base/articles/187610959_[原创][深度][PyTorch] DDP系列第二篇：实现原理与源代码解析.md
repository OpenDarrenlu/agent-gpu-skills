# [原创][深度][PyTorch] DDP系列第二篇：实现原理与源代码解析

**作者**: 小志哥​​上海引望智能技术有限公司 员工

**原文链接**: https://zhuanlan.zhihu.com/p/187610959

---

https://medium.com/@esaliya/model-parallelism-in-deep-learning-is-not-what-you-think-94d2f81e82ed
概览

想要让你的PyTorch神经网络在多卡环境上跑得又快又好？那你definitely需要这一篇！

No one knows DDP better than I do!
– – MagicFrog（手动狗头）

本文是DDP系列三篇（基本原理与入门，实现原理与源代码解析，实战与技巧）中的第二篇。本系列力求深入浅出，简单易懂，猴子都能看得懂（误）。本篇主要聚焦于DDP原理和源代码解析。

虽然是进阶篇，但是本篇力求做到简单易懂，涉及的新概念都会有讲解、引用。看完这篇后，你的收获将是：

了解分布式计算的概念
了解PyTorch模型的状态表示和构成
学习DDP的精巧的实现技巧
学会如何debug你的DDP模型

请欢快地开始阅读吧！

依赖

pytorch(gpu)>=1.5，python>=3.6

复习

我们先回顾一下DDP的代码实现。如果有看不懂的地方，那就得回去看上一篇哦。

################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 假设我们的数据是这个
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform)
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
model = ToyModel().to(local_rank)
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)


################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
背景概念

在正式介绍之前，我们先认识一些基本概念，打好基础。地基是很重要的，请各位同学认真学习哦！

分布式编程

一个分布式系统，相对于单机系统，其最大的特征就是，其数据、处理是分布在不同地方的。与此相伴的是，各节点间有交换数据的需求，为此需要定义交换数据的规范、接口。在此基础上，才能构建起分布式计算的大框架。比如很有名的google大数据三驾马车之一的`map-reduce`概念，简要地描述，就是将数据分开成N份map到N个地方，并行进行处理；处理完成后，再将结果reduce到一起。

为了满足分布式编程的需求，PyTorch提供了一些分布式基本接口，在torch.distributed中。有兴趣的可以自己翻阅：文档 and 代码

下图阐述了PyTorch实现的分布式接口：

记住我们使用的是最常用的NCCL后端，是GPU上优化做得最好的后端。

在DDP这里，我们重点介绍一下最重要的实现，all_reduce。

所谓的reduce，就是不同节点各有一份数据，把这些数据汇总到一起。在这里，我们规定各个节点上的这份数据有着相同的shape和data type，并规定汇总的方法是相加。简而言之，就是把各个节点上的一份相同规范的数据相加到一起。
所谓的all_reduce，就是在reduce的基础上，把最终的结果发回到各个节点上。
具体的allreduce实现，要看具体的backend。流行的GPU backend NCCL，all_reduce的实现就是使用了ring思想。

DDP利用all_reduce，来进行不同进程上的梯度的平均操作。PyTorch提供了几个all_reduce的版本，下面这个就是Ring-Reduce版本（我们在前篇阐述了为什么Ring-Reduce是一个更好的版本）：

def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=group.WORLD,
               async_op=False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.
    After the call ``tensor`` is going to be bitwise identical in all processes.
    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
PyTorch 数据结构基础

DDP到底和什么数据结构打交道呢？我们要首先解决这些问题：

我们知道，DDP下各进程不同步参数而是同步参数的变化量，所以各进程的模型的状态同一性是非常重要的。那么模型的状态由什么构成呢？
DDP是怎么做到，无论是什么模型进来，一个简单的model = DDP(model)就可以解决问题呢？它的逻辑是怎么嵌入到模型中的？




buffer

解决第一个问题，需要了解buffer的概念。
在PyTorch中，所有的模型都会继承module类。可以说，一个CNN模型，其就是由一系列module组合而成的。要了解模型，就必须从module下手。下面是module的初始化代码，可以看到，它定义了一系列变量。可以说，这些变量就组成了一个module的基本要素。

代码

# torch.nn.modules.py. line 71. Class module:
    def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()

总的来说，module的基本要素可以分为2组，一组是状态，一组是各种各样的hooks。状态有以下4个东西：

self.training
指的是网络是否在训练状态中。这是个非常宏观的状态，大家都知道这个是啥，可以略过。
self._modules
modules是下属的模块，相当于迭代地定义了self.trainig, self._modules, self._parameters等一系列变量
self._parameters
指的就是网络的参数
self._buffers
不是参数，但也对网络很重要，会被持久化保存的数据。
举个例子，BatchNorm中的moving mean and variance就是buffer，其优化不是通过梯度反向传播而是通过其他途径。

从本质上讲，当一个模型的网络结构被定义后，其状态就是由parameter和buffer的迭代组合表示的。当我们保存模型，调用model.staic_dict()的时候，我们同时会得到模型的parameter和buffer；也就是说，在DDP中，如果我们要在不同进程中维持相同的状态，我们不光要传递parameter的梯度，也要传递buffer。事实上，DDP就是这么做的。当每次网络传播开始前，其都会把master节点上的buffer广播给其他节点，维持状态的统一。




hook

回答第二个问题，需要了解hook的概念。
hook的中文是`钩子`，是一种技术概念。用形象的话讲，hook提供了这么一种机制：程序提供hook接口，用户可以写一个hook函数，然后钩在hook接口，即程序的主体上从而可以插入到中间执行。DDP使用hook技术把自己的逻辑插入到module的训练过程中去。

在前一篇文章中，曾经讲过

在模型训练时，各个进程通过一种叫Ring-Reduce的方法与其他进程通讯，从而获得所有进程的梯度；

那么，Ring-Reduce机制是怎么插入到module中去的呢？这归功于PyTorch提供了很多个hook接口！
其中，就有一个是，parameter在反向梯度计算结束后提供了一个hook接口。DDP把Ring-Reduce的代码写成一个hook函数，插入到这里。每次parameter的反向梯度计算结束后，程序就会调用这个hook函数，从而开启Ring-Reduce流程。因为所有模型都用到parameter，所以DDP模型用hook函数就解决了所有模型的梯度平均问题了！

下面，我们来看看其具体的代码实现

torch.nn.parameter

torch.nn.parameter只是torch.Tensor上的一层概念封装，没什么时候特别的。hook机制也是定义在torch.Tensor中的。

torch.tensor.Tensor

有一点需要说明，DDP的关键代码（即梯度平均）是用C++实现的。但是，在C++、python代码中Tensor都给出了hook接口，实现相似的功能。所以我们可以看下Tensor的python hook接口的文档，来理解下hook这个概念。

# line 200. Class Tensor.
    def register_hook(self, hook):
        r"""Registers a backward hook.
        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::
            hook(grad) -> Tensor or None
        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.
        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        Example::
            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v.grad
             2
             4
             6
            [torch.FloatTensor of size (3,)]
            >>> h.remove()  # removes the hook
        """
DDP内部实现

Finally，经过一系列铺垫，终于要来讲DDP是怎么实现的了。在读到这里的时候，你应该对DDP的大致原理、PyTorch是怎么训练的有一定的了解。现在就来了解一下最底层的细节吧！
下面，我们会给出具体源代码的URL，复习一下不同的DDP模式，给出一份DDP训练流程的伪代码，最后总结一下易错的注意事项。

代码位置

DDP的代码主要在以下几个地方：

https://github.com/pytorch/pytorch/blob/v1.5.0/torch/nn/parallel/distributed.py

https://github.com/pytorch/pytorch/blob/v1.5.0/torch/distributed/distributed_c10d.py

https://github.com/pytorch/pytorch/blob/v1.5.0/torch/csrc/distributed/c10d/reducer.h

同时推荐一个官方设计笔记，讲得很详细，有兴趣可以看看。

DDP模式

之前我们介绍过DDP模式。在这里，我们复习一下。因为，在接下来的DDP流程介绍中，我们是要处理不同的模式的。

1. 每个进程一张卡。这是DDP的最佳使用方法。
2. 每个进程多张卡，复制模式。一个模型复制在不同卡上面，每个进程都实质等同于DP模式。这样做是能跑得通的，但是，速度不如上一种方法，一般不采用。
3. 每个进程多张卡，并行模式。一个模型的不同部分分布在不同的卡上面。例如，网络的前半部分在0号卡上，后半部分在1号卡上。这种场景，一般是因为我们的模型非常大，大到一张卡都塞不下batch size = 1的一个模型。
正篇！正篇！DDP流程的伪代码

我们总结了一个DDP模型在训练过程中的伪代码，来清晰地描述DDP的细节。
DDP很简单，但是流程并不简单。额外的代码主要是在，处理不同的DDP模式以及加速。刨去这些，主体其实是很简单的，所以不要害怕，大胆看完！

准备阶段

环境准备（就是init_process_group这一步）。各个进程会在这一步，与master节点进行握手，建立连接。
注释：如果连接上的进程数量不足约定的 word_size，进程会一直等待。也就是说，如果你约定了world_size=64，但是只开了6台8卡机器，那么程序会一直暂停在这个地方。
DDP初始化（也就是model = DDP(model)这一步）
把parameter，buffer从master节点传到其他节点，使所有进程上的状态一致。
注释：DDP通过这一步保证所有进程的初始状态一致。所以，请确保在这一步之后，你的代码不会再修改模型的任何东西了，包括添加、修改、删除parameter和buffer！
（可能）如果有每个节点有多卡，则在每张卡上创建模型（类似DP）
把parameter进行分组，每一组称为一个bucket。临近的parameter在同一个bucket。
注释：这是为了加速，在梯度通讯时，先计算、得到梯度的bucket会马上进行通讯，不必等到所有梯度计算结束才进行通讯。后面会详细介绍。
创建管理器reducer，给每个parameter注册梯度平均的hook。
注释：这一步的具体实现是在C++代码里面的，即reducer.h文件。
（可能）为可能的SyncBN层做准备

正式训练阶段

在每个step中，DDP模型都会做下面的事情：

采样数据，从dataloader得到一个batch的数据，用于当前计算（for data, label in dataloader）。
注释：因为我们的dataloader使用了DistributedSampler，所以各个进程之间的数据是不会重复的。如果要确保DDP性能和单卡性能一致，这边需要保证在数据上，DDP模式下的一个epoch和单卡下的一个epoch是等效的。
进行网络的前向计算（prediction = model(data)）
同步各进程状态
（可能）对单进程多卡复制模式，要在进程内同步多卡之间的parameter和buffer
同步各进程之间的buffer。
接下来才是进行真正的前向计算
（可能）当DDP参数find_unused_parameter为true时，其会在forward结束时，启动一个回溯，标记出所有没被用到的parameter，提前把这些设定为ready。
注释：find_unused_parameter的默认值是false，因为其会拖慢速度。
计算梯度（loss.backward()）
reducer外面：各个进程各自开始反向地计算梯度。
注释：梯度是反向计算的，所以最后面的参数反而是最先得到梯度的。
reducer外面：当某个parameter的梯度计算好了的时候，其之前注册的grad hook就会被触发，在reducer里把这个parameter的状态标记为ready。
reducer里面：当某个bucket的所有parameter都是ready状态时，reducer会开始对这个bucket的所有parameter都开始一个异步的all-reduce梯度平均操作。
注释：
bucket的执行过程也是有顺序的，其顺序与parameter是相反的，即最先注册的parameter的bucket在最后面。
所以，我们在创建module的时候，请务必把先进行计算的parameter注册在前面，后计算的在后面。不然，reducer会卡在某一个bucket等待，使训练时间延长！
所谓的参数注册，其实就是创建网络层。也就是要求按照网络计算顺序，依次创建网络层。
reducer里面：当所有bucket的梯度平均都结束后，reducer才会把得到的平均grad结果正式写入到parameter.grad里面。
注释：这一步，感觉没有必要等全部结束之后才进行。可能得对照一下源码。
优化器optimizer应用gradient，更新参数（optimizer.step()）。
注释：这一步，是和DDP没关系的。

虽然DDP的实现代码与optimizer没有关系，但是关于optimizer有个额外的东西需要说明。更新后的参数最终能在各进程间保持一致，是由以下因素保证的：

参数初始值相同
参数更新值相同
更新值相同又是由以下因素保证的：
optimizer初始状态相同
每次opimizer.step()时的梯度相同。

我们可以看到，因为optimizer和DDP是没有关系的，所以optimizer初始状态的同一性是不被DDP保证的！
大多数官方optimizer，其实现能保证从同样状态的model初始化时，其初始状态是相同的。所以这边我们只要保证在DDP模型创建后才初始化optimizer，就不用做额外的操作。但是，如果自定义optimizer，则需要你自己来保证其统一性！

回顾一下文章最开始的代码，你会发现，optimizer确实是在DDP之后定义的。这个时候的模式已经是被初始化为相同的参数，所以能够保证优化器的初始状态是相同的。

# 新增：构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 优化器：要在构造DDP model之后，才能初始化model。
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
为什么速度没怎么提升/性能下降

很多同学可能有这么一个问题，我加入了DDP，为什么速度没怎么提升/性能下降了呢？我给大家准备了一个check list。

是否遵循了“单进程单卡”这样的最佳工程实践？
“单进程多卡复制模式”在速度上不是最优的，而且不被PyTorch社区优先支持，避免使用。
是否使用了默认的NCCL后端？
用就完事。
各进程的模型是否相同？
用户必须保证，不同进程里的模型都是相同结构的；保证parameter（你可以理解为网络层）的创建顺序是一致的。
模型的parameter创建顺序是否与真实计算顺序一致？
这涉及到bucket的通讯效率优化
产生DDP模型后，是否手动动了它的参数？
不允许在产生DDP后，新增、减少、随机修改、替换参数，会造成梯度reduce出错、各进程间的参数不相同、丢失hook机制。
DDP模式下的一个epoch的数据和单卡下的一个epoch的数据是否是等效的？
实际上，n卡的DDP模式，理论上可以等价于n次gradient accumulation的单卡模式。所以，确保你的数据，也是这样的。
如果出现性能下降，切记数据是最有可能出现问题的地方！
是否保证初始状态的同一性？
parameter、buffer初始状态同一性
optimizer初始状态同一性
DistributedSampler机制

最后，我们额外介绍一下DDP的DistributedSampler机制。

不知道你有没有好奇，为什么给dataloader加一个DistributedSampler，就可以无缝对接DDP模式呢？其实原理很简单，就是给不同进程分配数据集的不重叠、不交叉部分。那么问题来了，每次epoch我们都会随机shuffle数据集，那么，不同进程之间要怎么保持shuffle后数据集的一致性呢？DistributedSampler的实现方式是，不同进程会使用一个相同的随机数种子，这样shuffle出来的东西就能确保一致。

具体实现上，DistributedSampler使用当前epoch作为随机数种子，从而使得不同epoch下有不同的shuffle结果。所以，记得每次epoch开始前都要调用一下sampler的set_epoch方法，这样才能让数据集随机shuffle起来。

下面看一下DistributedSampler的核心源代码：

代码

# line 56
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
# line 79
    def set_epoch(self, epoch):
        self.epoch = epoch
总结

既然看到了这里，不妨点个赞/喜欢吧！

在本篇中，我们详细介绍了DDP的原理和底层代码实现。如果你能完全理解，相信你对深度学习中的并行加速、分布式计算会有更深入的认识。知己知彼，方能百战不殆，对DDP有透彻的了解，才能让你的模型以最快的速度跑起来，加快实验迭代速度，极大地提高产出！

但是，正所谓理论联系实践，如果只掌握理论而不进行实践，无疑是纸上谈兵。代码的有趣地方也是在这里，就算代码设计得再好，理解得再透彻，实际编程过程中，你还是会发现遍地是坑。笔者有幸踩过一些坑，跟大家分享一下。请各位有志之士阅读DDP系列第三篇：实战！
