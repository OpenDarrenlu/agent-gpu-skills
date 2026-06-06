# tensorrt 实践小结一：onnx-parser

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/718348437

---

​
目录
收起
环境准备
tensorrt 安装
tensorrt OSS安装
关于 onnx
onnx-parser
环境准备

安装 tensorrt 库和 tensorrt OSS，可略过~

tensorrt 安装

到 https://developer.nvidia.com/tensorrt/download/10x 下载符合自己硬件环境的版本, 如：

wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/local_repo/nv-tensorrt-local-repo-ubuntu2004-10.4.0-cuda-12.6_1.0-1_amd64.deb
dpkg  -i nv-tensorrt-local-repo-ubuntu2004-10.4.0-cuda-12.6_1.0-1_amd64.deb
cp /var/nv-tensorrt-local-repo-ubuntu2004-10.4.0-cuda-12.6/nv-tensorrt-local-A88B7455-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get install tensorrt
dpkg-query -W tensorrt #验证安装是否成功 
(dpkg -l | grep TensorRT #验证安装是否成功 )

安装成功后，库在 /usr/lib/x86_64-linux-gnu 路径下，头文件在 /usr/include/x86_64-linux-gnu，此外 /usr/src/tensorrt/ 还有一些测试学习的demo。

tensorrt OSS安装

tensorrt oss 主要是将 tensorrt 里面的 cpp api 绑定为 python api，暴漏 tensorrt 中开源的部分（主要是一些头文件的定义以及 plugin），以及重要的 onnx-parser 工具（将onnx模型转为trt network），外带一些可以学习的demo。

分支：release/10.4

commit 号：866548c95c6d7113a4c5f6a440022b2d216c04cf

git submodule update --init --recursive

# c++ api，tensorrt、parser、protobuf、plugin 等动态库编译
cd TensorRT
mkdir build && cd build
cmake .. -DTRT_OUT_DIR=`pwd`/out
make -j24

# python wheel 包编译
cd TensorRT/python
# 参考里面的 README.md 文档(需要指定刚刚编译完的parser等库，否则就默认使用/usr/lib/x86_64-linux-gnu路径下的库)
VERBOSE=ON TRT_OSSPATH={TRT_OSSPATH} TENSORRT_BUILD={TRT_OSSPATH}/build/output \
EXT_PATH={pybind_11_path} TENSORRT_MODULE=tensorrt \
PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=10 TARGET_ARCHITECTURE=x86_64 ./build.sh

# 增量编译
cd TensorRT/build && make -j # 重新编译 parser, plugin 等库
cd TensorRT/python/build && make -j  # 重新编译 tensorrt.so 
cp tensorrt/tensorrt.so bindings_wheel/tensorrt/tensorrt.so
cd bindings_wheel
python setup.py bdist_wheel #重新编译 wheel 包

这里插一句，

https://github.com/NVIDIA/TensorRT/blob/release/10.4/python/build.sh#L39

https://github.com/NVIDIA/TensorRT/blob/release/10.4/python/CMakeLists.txt#L114

编译脚本中最后一个参数应该写错了，改为传入 -DTENSORRT_BUILD=${TENSORRT_BUILD}，否则无法使用自己编译的onnx-parser等（默认使用/usr/lib/x86_64-linux-gnu路径下下载的）。

修改 build.sh 脚本

再插一句，为了方便阅读源码和快速跳转定位，让 VERBOSE log 也显示出代码的具体行，可以改下代码。

比如直接把 if 判断去掉。

关于 onnx

下面两个博客可以参考下，也建议阅读下 onnx 的 proto 定义，定义是简单明了的~

进击的程序猿：ONNX 模型分析与使用

BBuf：ONNX学习笔记

onnx-parser

onnx-parser 主要是将 onnx 模型转换为 tensorrt 的 IR（即 tensort 中的 network）。

该功能原理上非常简单，首先需要熟悉一下使用 c++ api 如何从头开始创建一个 network，实际上就是调用 addXXX 之类的 api 来在 network 中添加 layer，参考官方文档： create-network-def-scratch。

network 本质是一个也是计算图，主要有两种节点，一个是 op，在network 中叫 ILayer， 另外一种是输入和输出，也分为两种，一种是 Itensor， 一种是 Weights。

比如下面是增加 conv 算子的 api：

onnx 格式的模型有以下特点：

onnx 格式中，每个 Layer（Op）输出的 tensor 名字必须是唯一的
如果一个 tensor 是某个 Layer 的输出，它必然不是另外一个 Layer 的输入，否则就会形成环。
onnx 有一些 value 是 constant（initializers），这些 initializers 可能是某些 op 的权重，如 Linear， 或者 op 的某个输入是常量，如 matmul 的第二个输入是常量

从官方文档上看， NVIDIA Deep Learning TensorRT Documentation 只需要两行代码即可完成这个转换过程。

代码实现上看，该 python api 绑定了 https://github.com/onnx/onnx-tensorrt.git 仓库中的 onnx2trt::ModelImporter 对象。

并调用 parseFromFile API， parseFromFile 调用 importModel api，完成 onnx 模型到 network 的转换。

importModel 主要流程如下：

首先模型有很多输入，调用 addInput api，针对每个输入给 network 添加输入

2. 在 onnx 格式中，initializers 也算作模型的输入，不过不是 network 的输入，针对每一个 initializers，创建一个 Weights 对象，在 onnxpaser 的实现中，没有直接使用 tensorrt::Weights 对象，而是自己定义了一个 ShapedWeights 的类，这个类可以直接转成 tensorrt::Weights，以供 addxxxLayer api 使用。在构建 weights 对象过程中，会将 onnx 模型中的数据 copy 到 weights 对象中。

上面两个过程中，创建的 Itensor 和 ShapedWeights 会被保存在一个 map 中，能够根据名字索引到，以供后面 addLayer api 使用。

3. 接下来，获取算子的 topo 排序，然后根据不同的 onnx算子找到不同的转换函数，调用即可。

根据拓扑排序，一个一个的转换onnx算子
找到不同onnx算子对应的转换函数

转换函数的设计模式也很简单，如下，所有的转换函数都在 builtin_op_importers 中，每个 onnx 算子都有一个 Type，根据 Type 从注册的转换函数中，找到对应转换函数再调用即可。

所有的转换函数根据 op 的 Type，注册到了一个 map 数据结构中

注册转换函数使用宏定义 DEFINE_BUILTIN_OP_IMPORTER。如下是 conv 的转换函数：

将 onnx 中的 conv 算子，转换为 network 中的 ILayer 的转换函数

不同转换函数的过程也基本一致，首先找到该 Layer 的所有输入（Itensor 和 Weights），然后调用 addXXX 即可，然后设置一些该 Layer 的属性。

注意的是，前面第 2 点说过 initializers 都被转换为了 ShapedWeights --> Weights， 但是某些initializers 并不是权重，而是某个 op 的输入，即对应的 addXXX api要求输入的参数是 Itensor，而不是 Weights，这时候会add 一个 ConstLayer， 把这个 Weights 类型转为 Itensor

将weights转为Itensor以适配某些 addxxxLayer 的 api

4. 最后，add 一个 layer 之后，这个 layer 会有 output tensor，获取这些输出 Itensor，标记名字，和 input Itensor，Weights一样，保存在同一个 map 后，以供后面的 addxxxLayer api 使用。所有的 Layer 添加完之后，标记一下哪些 Itensor 是 network的输出。

将layer 的输出 Itensor 存储在同一个 map 中

5. 总结一下，onnx-parser 整体功能围绕着 addXXXLayer api 工作，设计模式上也清晰明了。Network 有大量的 addXXX 之类的 api 来添加 layer， 并都绑定了 python api

INetworkDefinition 中的 add layer api

这是一个简单的 demo，在构建了 network 之后，可以逐个 Layer 的遍历，打印每一个 layer 的信息。

import torch
import torch.nn as nn
import tensorrt as trt

class Network_Visualization():
    def __init__(self, newwork):
        self.network = network
        self.tensor_info_map = {}

    def print_topo_order(self):
        network = self.network
        t_map = self.tensor_info_map
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            string = f"{i} Type: {layer.type}".ljust(40)
            string += f"name: {layer.name}".ljust(30)
            string += "Inputs: "
            if layer.num_inputs == 0:
                string += "None; "
            else:
                for j in range(layer.num_inputs):
                    i_tensor = layer.get_input(j)
                    shape_dtype = "{" + f"{i_tensor.shape}, {i_tensor.dtype}" + "}"
                    string += f"{i_tensor.name}:{shape_dtype}; "
                    t_map[i_tensor.name] = shape_dtype
            string += "Outputs: "
            if layer.num_outputs == 0:
                string += "None; "
            else:
                for j in range(layer.num_outputs):
                    i_tensor = layer.get_output(j)
                    shape_dtype = "{" + f"{i_tensor.shape}, {i_tensor.dtype}" + "}"
                    string += f"{i_tensor.name}:{shape_dtype}; "
                    t_map[i_tensor.name] = shape_dtype
            print(string)


In_dim = 64
Out_dim = 64
class custom_model(torch.nn.Module):
    def __init__(self):
        super(custom_model, self).__init__()
        self.linear = nn.Linear(In_dim, Out_dim, dtype=torch.float16, bias=False)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.add(x, 1)
        return  x

model = custom_model().cuda()
batch = 16
x = torch.randn(size=(batch, In_dim), dtype=torch.float16, device="cuda")
torch.onnx.export(model,  
                args = (x),
                f = "test.onnx",
                input_names = ["x"],
                export_params = True,
                output_names = ["output"],)

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)

flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
flag = flag << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file("test.onnx")
assert success

vis_tool = Network_Visualization(network)
vis_tool.print_topo_order()

config = builder.create_builder_config()
# config.set_flag(trt.BuilderFlag.FP16)
serialized_engine = builder.build_serialized_network(network, config)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)

context = engine.create_execution_context()
out = torch.empty_like(x)
context.set_tensor_address("x", x.data_ptr())
context.set_tensor_address("output", out.data_ptr())
stream = torch.cuda.Stream()
context.execute_async_v3(stream.cuda_stream )

out_ref = model(x)
torch.cuda.synchronize()
assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-3)
逐层打印每个 Layer 的信息

动态 shape 的场景：

import torch
import torch.nn as nn
import tensorrt as trt

class Network_Visualization():
    def __init__(self, newwork):
        self.network = network
        self.tensor_info_map = {}

    def print_topo_order(self):
        network = self.network
        t_map = self.tensor_info_map
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            string = f"{i} Type: {layer.type}".ljust(40)
            string += f"name: {layer.name}".ljust(30)
            string += "Inputs: "
            if layer.num_inputs == 0:
                string += "None; "
            else:
                for j in range(layer.num_inputs):
                    i_tensor = layer.get_input(j)
                    shape_dtype = "{" + f"{i_tensor.shape}, {i_tensor.dtype}" + "}"
                    string += f"{i_tensor.name}:{shape_dtype}; "
                    t_map[i_tensor.name] = shape_dtype
            string += "Outputs: "
            if layer.num_outputs == 0:
                string += "None; "
            else:
                for j in range(layer.num_outputs):
                    i_tensor = layer.get_output(j)
                    shape_dtype = "{" + f"{i_tensor.shape}, {i_tensor.dtype}" + "}"
                    string += f"{i_tensor.name}:{shape_dtype}; "
                    t_map[i_tensor.name] = shape_dtype
            print(string)


In_dim = 64
Out_dim = 64
class custom_model(torch.nn.Module):
    def __init__(self):
        super(custom_model, self).__init__()
        self.linear = nn.Linear(In_dim, Out_dim, dtype=torch.float16, bias=False)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.add(x, 1)
        return  x

model = custom_model().cuda()
batch = 16
x = torch.randn(size=(batch, In_dim), dtype=torch.float16, device="cuda")

# 定义动态轴
dynamic_axes = {
    'x': {0: 'm'},  # 输入的第 0 轴是动态的
    'output': {0: 'm'}  # 输出的第 0 轴是动态的
}
torch.onnx.export(model,  
                args = (x),
                f = "test.onnx",
                input_names = ["x"],
                export_params = True,
                dynamic_axes = dynamic_axes,
                output_names = ["output"],)

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)

flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
flag = flag << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file("test.onnx")
assert success

vis_tool = Network_Visualization(network)
vis_tool.print_topo_order()

# 定义动态形状范围
profile = builder.create_optimization_profile()
profile.set_shape("x", (1, 64), (64, 64), (128, 64))

config = builder.create_builder_config()
config.add_optimization_profile(profile)
# # config.set_flag(trt.BuilderFlag.FP16)
serialized_engine = builder.build_serialized_network(network, config)

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
print(f"num_optimization_profiles: {engine.num_optimization_profiles}")

context = engine.create_execution_context()
stream = torch.cuda.Stream()
m = 127
context.set_input_shape("x", (m, 64))
context.set_optimization_profile_async(0, stream.cuda_stream)
inp = torch.randn(size=(m, In_dim), dtype=torch.float16, device="cuda")
out = torch.empty_like(inp)
context.set_tensor_address("x", inp.data_ptr())
context.set_tensor_address("output", out.data_ptr())
context.execute_async_v3(stream.cuda_stream )

out_ref = model(inp)
torch.cuda.synchronize()
assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-2)
包含动态shape时，每个 layer 的信息
