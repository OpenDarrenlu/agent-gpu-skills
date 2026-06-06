# 【OnnxRuntime】IR Pass（transform）梳理（一）

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/719060912

---

​
目录
收起
OnnxRuntime 计算图优化
GraphTransformer
RuleBasedGraphTransformer
可视化准备
EliminateIdentity
EliminateSlice
UnsqueezeElimination
EliminateDropout
ExpandElimination
CastElimination
PreShapeNodeElimination
NoopElimination
DivMulFusion
FuseReluClip
GemmSumFusion
GemmTransposeFusion
NotWhereFusion
ConvAddFusion
ConvMulFusion
ConvBNFusion
PadFusion
MatmulBNFusion
LabelEncoderFusion
小结
OnnxRuntime 计算图优化

在 ort 中，onnx 模型被加载后，转换为一个中间表达（IR），定义为 graph（即计算图）。（当然 onnx 也是一种 IR，是一个计算图，但是ort 没有直接在 onnx 模型上修改，而是将 onnx 模型转为了另一种 IR，ort框架内称为 graph）

https://github.com/microsoft/onnxruntime/blob/v1.19.2/include/onnxruntime/core/graph/graph.h
github.com/microsoft/onnxruntime/blob/v1.19.2/include/onnxruntime/core/graph/graph.h

graph 是一个严格的图定义，主要包含两种节点： Node 和 NodeArg， Node 就是算子（op）的抽象，NodeArg 就是算子输入变量（往往是tensor）的抽象，然后包含了Node之间的联系，即边 NodeEdge, NodeEdge 还包括了 EdgeEnd 等。本文不细谈 ort中 IR 的设计思想和实现技巧，主要关注在计算图上的优化。

https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/session/inference_session.cc#L1175
github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/session/inference_session.cc#L1175

从代码注释上出，图优化主要有七点：

第一点是关于一些“function”类型op的优化，一般模型不包含这种算子。

第二点是针对量化模型的，在ort这种基于计算图的推理引擎中，模型量化时会插入量化（Q）和反量化算子（DQ），为了统计算子的输入数值、计算 max 等，推理时需要删掉这些算子。

第三点 level 1 是一些与硬件无关的算子优化，一些常见的算子融合、常量折叠等，本blog后续会梳理一部分。

第四点是一些与硬件有关（不同 backend）的优化。

第五点 level 2 是针对 CPU EP的优化。

第六点是插入 cast 算子，一般是在给每个 op 选择了 kernel 后，遇到前后两个 kernel 输出输入精度不匹配时，会自动插入 cast 算子。

第七点是插入 copy 算子，针对异构计算场景， 如前一个算子只支持在 cpu 计算，后面的算子可以在 gpu 上计算，当前面算子计算完时，需要将其结果 copy 到 gpu。

优化计算图的类，ort 中都是从 GraphTransformer 基类派生来的（本文中 也叫pass），图优化 pass 会跑多次，直到计算图不再发生任何改变。

代码如下:

steps_ 是预先设置的（源码中设置为10，已经足够大了），跑的过程中有些 pass 只能跑一次，跑过了的话就跳过，最后直到整个计算图不再发生任何改变了，这时就认为图优化已经充分了。

这样设计的目的也很清晰，每一个pass都可能会改变计算图，第一轮时，前面某个pass没有匹配上，在经过了后面的pass修改了图之后，第二轮循环时前面的pass就可能被匹配到并触发。

GraphTransformer

这是所有 pass 的基类，派生类需要重定义 ApplyImp 类。

RuleBasedGraphTransformer

这是 level 1 优化中重要的一个类，可以基于改写规则改写 graph， 改为规则由 RewriteRule 定义，这些改写规则是和硬件无关的，纯粹的 op 逻辑上的等价改写。后面会梳理里面的改写规则。

该 pass（RuleBasedGraphTransformer） 执行的整体逻辑：获得计算图 gprah 的拓扑排序，然后依次循环遍历每一个算子，根据每一个算子的名字，寻找是否有匹配的 RewriteRule，如果有，则尝试运行这个 RewriteRule。RewriteRule 执行前会进一步检测条件，由 SatisfyCondition 函数定义。

下面是对 level 1 中的改写规则的梳理。

level1改写规则
可视化准备

为了测试某个 pass 是否生效，可以用 torch 定义一个能够触发该 pass 的 model，然后使用 torch.jit.trace 转为 onnx 静态图，再使用 onnxruntime 运行， 如下：

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

M = 4
N = 16
K = 16
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=N, dtype=torch.float16, affine=True, track_running_stats=True)
        self.y = nn.Parameter(torch.randn(size=(K, N), dtype=torch.float16))
    
    def forward(self, x, ):
        out = torch.matmul(x, self.y)
        out = self.bn(out)
        return out


model = custom_model().cuda().eval()
x = torch.randn(size=(M, K), dtype=torch.float16, device="cuda")
out_ref = model(x)

# 基于jit.trace record 模型运行时跑过的算子， 将 torch 动态图转为静态计算图（onnx格式）， 使用opset_version = 17
torch.onnx.export(model, (x), "model.onnx", verbose=False, input_names=["x"], output_names=["output"], opset_version=17)

providers = [("CUDAExecutionProvider", {'enable_cuda_graph': False})]
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 1
sess_options.inter_op_num_threads = 1
sess_options.intra_op_num_threads = 1
ort_session = ort.InferenceSession("model.onnx", sess_options, providers=providers)

out_ref = model(x)
out = ort_session.run(output_names=["output"],
                    input_feed={
                                "x" : x.cpu().numpy()
                                })[0]
assert torch.allclose(out_ref, torch.tensor(out, device=out_ref.device), rtol=1e-3, atol=1e-5)


为了可视化 RuleBasedGraphTransformer pass 对计算图的修改，代码中需要将变化后的graph 保存为onnx格式，如下对 rule_based_graph_transformer.cc 做一点简单的修改：

将修改后的graph重新保存为onnx格式

onnxruntime 编译：

./build.sh --use_cuda --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=89 \
           --allow_running_as_root --cuda_home /usr/local/cuda --cudnn_home /usr/include/ \
           --parallel 4 --nvcc_threads 4 --config Release --skip_test --build_shared_lib \
           --build_wheel --enable_pybind 


# 增量编译并安装 wheel 包
cd ./build/Linux/Release
make -j && python ../../../setup.py  bdist_wheel --wheel_name_suffix=gpu
python -m pip install dist/onnxruntime_gpu-1.20.0-cp310-cp310-linux_x86_64.whl --force-reinstall --no-deps
EliminateIdentity

正如它的名字所意，它的作用是消除计算图中的 "Identity" 算子，Identity 算子是一个占位符，不做任何计算，直接返回它的输入, 相当于赋值操作，在实际导出 onnx 模型时，计算图可能不会有这个算子（导出的过程中被优化了）。

测试 model：

M = 4
N = 16
K = 16
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=N, dtype=torch.float16, affine=True, track_running_stats=True)

    def forward(self, x, ):
        out = self.bn(x)
        return out

该 pass 优化后的效果：

EliminateSlice

按照其说明，消除下面场景的 slice 算子: slice 前后 tensor 不发生改变。

此时 slice 后的 tensor 和原 tensor 完全一致，所以可以 remove，比较简单，跳过测试。

UnsqueezeElimination

消除 Unsqueeze 算子。从其触发条件上看，当且只当 Unsqueeze 算子的输入是常量时，则消除这个 unsqueeze 算子，使用新的 unsqueeze 之后的 constant 作为输入。因此这个 pass 类似常量折叠功能。

测试 model：

batch = 1
M = 4
N = 16
K = 16
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Parameter(torch.randn(size=(K, N), dtype=torch.float16, device="cuda"))

    def forward(self, x, ):
        y = self.y.unsqueeze(0)
        out = torch.matmul(x, y)
        return out

model = custom_model().cuda().eval()
x = torch.randn(size=(batch, M, K), dtype=torch.float16, device="cuda")
out_ref = model(x)

# 基于jit.trace record 模型运行时跑过的算子， 将 torch 动态图转为静态计算图（onnx格式）
torch.onnx.export(model, (x), "model.onnx", verbose=False, input_names=["x"], output_names=["output"], opset_version=7)

（opset_version=7， 高版本在导出的过程中就会被优化掉 unsqueeze 算子）

该 pass 优化后的效果：

EliminateDropout

这个也很简单，消除 dropout 算子， dropout 算子是训练时为了防止过拟合加入的，按照一定的几率，让一些数值变为0，推理时不需要。（实际上 torch 组网后的 model，调用 model.eval() 接口也可以删掉 dropout 算子）

ExpandElimination

当 expand 的输入也是 constant 时，则可以消去 expand 算子，用 expand 之后的 constant 输入替换（类似常量折叠功能）。

测试 model：

batch = 4
M = 4
N = 16
K = 16
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Parameter(torch.randn(size=(1, K, N), dtype=torch.float16, device="cuda"))

    def forward(self, x, ):
        y = self.y.expand(batch, K, N)
        out = torch.matmul(x, y)
        return out

model = custom_model().cuda().eval()
x = torch.randn(size=(batch, M, K), dtype=torch.float16, device="cuda")
out_ref = model(x)

该 pass 的优化效果：

CastElimination

和上述类似，遇到一些无意义的精度转换时，如 tensor 本身是 float16， cast 又转为 float16，则是无意义的操作， 则可 remove cast 算子。

测试 model：

M = 4
N = 16
K = 16
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.to(torch.float16)
        out = torch.add(x, 1)
        return out

model = custom_model().cuda().eval()
x = torch.randn(size=(M, K), dtype=torch.float16, device="cuda")
out_ref = model(x)

该 pass 优化后的效果：

PreShapeNodeElimination

这个 pass 非常奇怪，删除 shape 前面的 cast 算子？ 构建了一个 graph，测试了下，最后的融合结果不太符合逻辑。

import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort

M = 16
N = 16
# 创建一个输入张量 X
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [M, N])

# 创建 Cast 算子，将输入张量 X 转换为 INT32 类型
cast_node = helper.make_node(
    'Cast',  # 算子名称
    inputs=['X'],  # 输入
    outputs=['cast_X'],  # 输出
    to=TensorProto.FLOAT16  # 目标数据类型
)

# 创建 Shape 算子，获取转换后的张量 cast_X 的形状
shape_node = helper.make_node(
    'Shape',  # 算子名称
    inputs=['cast_X'],  # 输入
    outputs=['shape']  # 输出
)

# 创建输出张量
shape_output = helper.make_tensor_value_info('shape', TensorProto.INT64, [2])

# 创建图
graph = helper.make_graph(
    [cast_node, shape_node],  # 节点列表
    'cast_shape_graph',  # 图的名称
    [X],  # 输入列表
    [shape_output]  # 输出列表
)

# 创建模型
model = helper.make_model(graph, producer_name='cast-shape-example', opset_imports=[helper.make_opsetid("", 17)])

# # 创建模型
# model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

# 保存模型到文件
onnx.save(model, 'cast-shape-example.onnx')

providers = [("CUDAExecutionProvider", {'enable_cuda_graph': False})]
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 1
sess_options.inter_op_num_threads = 1
sess_options.intra_op_num_threads = 1
ort_session = ort.InferenceSession("cast-shape-example.onnx", sess_options, providers=providers)

最后 cast 和 shape 都被删了。有懂的大佬可以讲一讲。

NoopElimination

消除一些不影响数值的加减乘除运算，如加减0，乘除1。

比较简单，跳过模型测试。

DivMulFusion

将连续的 Div 和 Mul 两个算子融合为一个 Div 算子，但是要求第一个 Div 算子的被除数是 1。

如一下 demo：

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np


In_Dim = 64
Out_Dim = 64
batch = 4
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.div(1, x)
        x = torch.mul(x, 3)
        return x

model = custom_model().cuda()
inp = torch.randn(size=(batch, In_Dim), dtype=torch.float16, device="cuda")
out_ref = model(inp)

触发条件很严格，从数学逻辑上来说，可以放的更宽一些。

该 pass 的优化效果：

DivMulFusion pass 优化前后对比
FuseReluClip

融合 Relu 和 Clip 算子， relu 定义如下: ReLU(x) = max(0, x)

clip 操作将 tensor 的数值限制在 (min, max) 之间，二者可以融合为 clip(x, min=0, max=max)。

如以下model：

class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = torch.clip(x, min=-1, max=10)
        return x

优化前后的计算图如下：

FuseReluClip pass 优化前后对比
GemmSumFusion

融合 gemm + sum 算子，在 blas 中 gemm 定义： D=\alpha A*B+\beta C , 因此可做如下融合（把加数 C 融合到 gemm 的 C 中）：

注意这里的 Sum 是逐个元素相加，相当于 elementwise-add（torch.add），torch 里的sum是对 tensor 元素求和，下面手动构建一个 onnx graph来理解和验证这个 pass。

import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort

M = 16
K = 16
N = 16

# 创建输入矩阵 A 和 B
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [M, K])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [M, N])

# 创建输入矩阵 C
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [M, N])

# 创建 Gemm 操作符
gemm_node = helper.make_node(
    'Gemm',
    inputs=['A', 'B'],
    outputs=['Y'],
    alpha=1.0,
    beta=0.0,
    transA=0,
    transB=0
)

# 创建 Sum 操作符
sum_node = helper.make_node(
    'Sum',
    inputs=['Y', 'C'],
    outputs=['Z']
)

# 创建输出矩阵 Z
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [M, N])

# 创建图
graph = helper.make_graph(
    [gemm_node, sum_node],
    'gemm_sum_example',
    [A, B, C],
    [Z]
)

# 创建模型
model = helper.make_model(graph)

# 保存模型到文件
onnx.save(model, 'gemm_sum_model.onnx')

print("Model saved to gemm_sum_model.onnx")

providers = [("CUDAExecutionProvider", {'enable_cuda_graph': False})]
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 1
sess_options.inter_op_num_threads = 1
sess_options.intra_op_num_threads = 1
ort_session = ort.InferenceSession("gemm_sum_model.onnx", sess_options, providers=providers)

优化前后计算图对比如下：

GemmTransposeFusion

融合 gemm 和 transpose 算子，变成一个 gemm 算子，因为 blas 库中， gemm 可以设定输入是否是 transpose，即使对 gemm 的输出做 transpose 也可以融合，因为： （AB）^ T=B^T*A^T ，此时特别注意 矩阵 C 是空输入（比如定义 Linear 时，就不能有 bias）。

验证 model：

(torch 定义 Linear 时，如果定义了 bias（Wx+b=AB+C），导出来的就是 gemm算子，如果没有 bias（Wx = AB），导出来就是 matmul 算子，matmul 算子和该改写规则不匹配，所以下面通过onnx graph，显示创建一个 C 为空的 gemm 算子)

import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort

M = 16
K = 16
N = 16
# 创建输入矩阵 A 和 B
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [M, K])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [M, N])

# 创建 Gemm 操作符
gemm_node = helper.make_node(
    'Gemm',
    inputs=['A', 'B'],
    outputs=['Y'],
    alpha=1.0,
    beta=0.0,
    transA=0,
    transB=0
)

# 创建 Transpose 操作符
transpose_node = helper.make_node(
    op_type='Transpose',
    inputs=['Y'],
    outputs=['Z'],
    perm=[1, 0] 
)

# 创建输入矩阵 C
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [N, M])
# 创建 Sum 操作符
sum_node = helper.make_node(
    'Sum',
    inputs=['Z', 'C'],
    outputs=['O']
)

# 创建输出矩阵 O
O = helper.make_tensor_value_info('O', TensorProto.FLOAT, [N, M])

# 创建图
graph = helper.make_graph(
    [gemm_node, transpose_node, sum_node],
    'gemm_transpose_sum_example',
    [A, B, C],
    [O]
)

# 创建模型
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

# 保存模型到文件
onnx.save(model, 'gemm_transpose_sum_example.onnx')
print("Model saved to gemm_transpose_sum_example.onnx")

providers = [("CUDAExecutionProvider", {'enable_cuda_graph': False})]
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 1
sess_options.inter_op_num_threads = 1
sess_options.intra_op_num_threads = 1
ort_session = ort.InferenceSession("gemm_transpose_sum_example.onnx", sess_options, providers=providers)

该 pass 优化前后对比图：

该graph先触发了 gemm 和 transpose 融合，然后 gemm 和 sum 融合。

NotWhereFusion

Not 算子和 Where 算子融合为一个 Where 算子，逻辑如下：

测试 model：

M = 16
K = 16
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, condition, x, y):
        condition = ~condition
        out = torch.where(condition, x, y)
        out = torch.add(out, 1)
        return out

model = custom_model().cuda().eval()
x = torch.randn(size=(M, K), dtype=torch.float16, device="cuda")
y = torch.randn(size=(M, K), dtype=torch.float16, device="cuda")
condition = torch.tensor([True], dtype=bool, device="cuda")
out_ref = model(condition, x, y)

# 基于jit.trace record 模型运行时跑过的算子， 将 torch 动态图转为静态计算图（onnx格式）
torch.onnx.export(model, (condition, x, y), "model.onnx", verbose=False, input_names=["condition", "x", "y"], output_names=["output"], opset_version=9)

该 pass 的融合效果：

ConvAddFusion

将 conv 后面的 add 算子融合到 conv 的 bias 中。

conv 的输入 shape 是（C, H, W)， 输出 shape 是（O_C, O_H, O_W), 其中 conv 的 bias 权重 shape 为（O_C,1，1，即每个output_channel 上， 加上一个 bias 数值。因此，后面的 add 算子必须和 bias 权重有相同的 shape 且为 add 的数值为常数（constant value）时，才能触发融合条件。

测试 model：

batch = 3
C = 3
H = 16
W = 16
Out_C = 4
Out_H = 14
Out_W = 14
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=C, out_channels=Out_C, kernel_size=(3, 3), stride=1, dtype=torch.float16, bias=True)
        self.add = nn.Parameter(torch.randn(size=(Out_C, 1, 1), dtype=torch.float16))

    def forward(self, x):
        x = self.conv(x)
        out = torch.add(x, self.add)
        return out

融合后效果如下：

ConvMulFusion

和 ConvAddFusion 触发条件类似，后面 mul 算子的输入的shape 必须为（O_C, 1,1）且为常量tensor时，才能将 mul 的数值融合到 conv 的 weight 中。

测试 model：

batch = 3
C = 3
H = 16
W = 16
Out_C = 4
Out_H = 14
Out_W = 14
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=C, out_channels=Out_C, kernel_size=(3, 3), stride=1, dtype=torch.float16, bias=True)
        self.mul = nn.Parameter(torch.randn(size=(Out_C, 1, 1), dtype=torch.float16))

    def forward(self, x):
        x = self.conv(x)
        out = torch.mul(x, self.mul)
        return out

融合后效果如下：

ConvBNFusion

conv 算子和 batchNorm 算子融合，融合原理：

卷积： z=w*x+b

BN： y=(z-mean)/\sqrt{var}*\beta + \gamma

合并两个式子，融合后的新卷积为：

w' =w/\sqrt{var}*\beta

b'=(b-mean)\sqrt{var}*\beta+\gamma

新的卷积就直接顺路完成 BN 的工作。

测试model：

batch = 3
C = 3
H = 16
W = 16
Out_C = 4
Out_H = 14
Out_W = 14
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=C, out_channels=Out_C, kernel_size=(3, 3), stride=1, dtype=torch.float16, bias=True)
        self.batchnorm = nn.BatchNorm2d(num_features=Out_C, dtype=torch.float16)

    def forward(self, x):
        x = self.conv(x)
        out = self.batchnorm(x)
        return out


model = custom_model().cuda().eval()
x = torch.randn(size=(batch, C, H, W), dtype=torch.float16, device="cuda")
out_ref = model(x)

# 基于jit.trace record 模型运行时跑过的算子， 将 torch 动态图转为静态计算图（onnx格式）
torch.onnx.export(model, (x), "model.onnx", verbose=False, input_names=["x"], output_names=["output"], opset_version=8)

（为了复现这个 pass 的作用，导出onnx静态图时设置 opset_version=8，过高的版本导出的过程中会自动融合）

融合后效果如下：

PadFusion

融合 pad 算子后后面的 conv， pool 等算子：

conv，pool 等算子，输入(H，W)时，在某些 stride 和 （k_h, k_w）组合下，最后一次滑动需要 pad 才能和卷积核相乘，相当于算子自带 pad 功能，当和前面的 pad 算子功能重合时（前面的pad变得冗余，conv算子内部会自己做pad），可以 remove pad 算子。

测试 model：

batch = 3
C = 3
H = 5
W = 5
Out_C = 4
Out_H = 3
Out_W = 3
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=C, out_channels=Out_C, kernel_size=(2, 2), stride=2, dtype=torch.float16, bias=True)
    
    def forward(self, x):
        x = nn.functional.pad(x, (1, 1, 0, 0), mode='constant', value=0)
        x = self.conv(x)
        return x


model = custom_model().cuda().eval()
x = torch.randn(size=(batch, C, H, W), dtype=torch.float16, device="cuda")
out_ref = model(x)

# 基于jit.trace record 模型运行时跑过的算子， 将 torch 动态图转为静态计算图（onnx格式）
torch.onnx.export(model, (x), "model.onnx", verbose=False, input_names=["x"], output_names=["output"], opset_version=8)

融合后效果如下:

MatmulBNFusion

将 matmul 算子和 batchnorm 算子融合为 gemm 算子，和 conv + batchnorm 融合类似，既然是融合为 gemm，则要求 matmul 中 tensor B 是 constant。同时要求 batchNorm 使用静态的均值和方差（非动态计算的）。

matmul 和 batchNorm 算子之间的存在一些不改变数值的操作（reshape， transpose），也不影响这两算子融合。

测试 model：

M = 4
N = 16
K = 16
class custom_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=N, dtype=torch.float16, affine=True, track_running_stats=True)
        self.linear = nn.Linear(in_features=K, out_features=N, dtype=torch.float16, bias=False)
    
    def forward(self, x, ):
        out = self.linear(x)
        out = self.bn(out)
        return out

model = custom_model().cuda().eval()
x = torch.randn(size=(M, K), dtype=torch.float16, device="cuda")
out_ref = model(x)

# 基于jit.trace record 模型运行时跑过的算子， 将 torch 动态图转为静态计算图（onnx格式）
torch.onnx.export(model, (x), "model.onnx", verbose=False, input_names=["x"], output_names=["output"], opset_version=17)

融合的效果：




LabelEncoderFusion

LabelEncoder 不是 torch 的标准算子，是 scikit-learn 中的一个将标签转成为数字的库，运算比较简单（cpu处理即可），输入要求是 numpy 类型，不需要使用 gpu，跳过。

小结

RuleBasedGraphTransformer 中 level 1 的改写规则就是上面梳理的这些，都是基础的与硬件无关的优化，这些优化在其他的推理框架上也都是存在的（如trt， paddle-inference），实现手段不同，但是目的都是相同的。实际模型推理时，如果没有触发这些pass，检测下 SatisfyCondition 函数的实现即可，一般是算子逻辑无法做等价转换或者op 的 version 不匹配等。
