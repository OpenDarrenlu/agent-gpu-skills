# R329 AIPU 初体验：ShuffleNet 编译到部署

**作者**: SunnyCase​DL Compiler

**原文链接**: https://zhuanlan.zhihu.com/p/390695615

---

前言

全志 R329 是 ARM China 周易 AIPU 落地的第一款芯片，最近刚好看到有 R329 开发板的免费申请活动 便迫不及待地申请了 SDK 来测试。本文记录了 ShuffleNet 在 R329 上的编译到部署的整个流程，帮助新玩家尽量减少一些踩坑吧～

准备开发环境
AIPU SDK，参考前言里的链接申请一个吧
Linux，AIPU SDK 只提供了 Linux 版本，我选择 Ubuntu 20.04
Python 3.6，AIPU SDK 只有 Python 3.6 版本，我选择 Anaconda
Git，从我的 GitHub clone 本文需要的代码和数据
配置开发环境
Anaconda

因为 AIPU SDK 需要 Python 3.6，为了不影响系统的其他东西，我们最好准备一个全新的 Python 虚拟环境，这里我选择 Anaconda。

wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sh Anaconda3-2021.05-Linux-x86_64.sh

按照提示下载并安装 Anaconda。我安装到的路径是 /home/sunnycase/anaconda3 安装完后我们需要使用 conda init把 conda 加到 PATH，我用的 shell 是 zsh，所以用下面的命令

/home/sunnycase/anaconda3/bin/conda init zsh
source ~/.zshrc
Python 3.6

下面我们创建 Python3.6 的虚拟环境，就起名叫 r329 吧。

conda create -n r329 python=3.6

根据提示安装完毕后我们通过下面的命令进入这个虚拟环境

conda activate r329

用完之后要退出就用

conda deactivate

另外为了加速 Pip 包的安装，我们可以采用清华源

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

最后我们安装本文需要的一些 Python 包

conda install opencv numpy
pip install onnx onnxruntime onnx-simplifier pyyaml
获取代码和模型

本文需要的代码、模型和数据以及预编译的AIPU程序等都放在我的 GitHub 上，我们先 Clone 到本地

git clone https://github.com/sunnycase/r329-test.git
配置 SDK

申请到的 SDK 我们只需要其中两部分 - AI610-SDK-1003-r0p0-eac0 里的内容复制到 r329-test/sdk/build 这个是我们用来编译和仿真模型的入口程序 - AI610-SDK-1002-r0p0-eac0 里的内容复制到 r329-test/sdk 这个是 AIPU 的仿真器

最终的目录结构是这样

(r329)  sunnycase@WorkS-1  /mnt/home-nas/repo/r329-test   master  tree -d
.
├── sdk
│   ├── build
│   │   ├── build-tool
│   │   ├── customized-op-example
│   │   │   ├── config
│   │   │   ├── input
│   │   │   ├── output_ref
│   │   │   └── plugin
│   │   └── user-case-example
│   │       ├── caffe
│   │       │   ├── config
│   │       │   ├── inception_v3_model
│   │       │   └── output_ref
│   │       ├── onnx
│   │       │   ├── config
│   │       │   ├── output_ref
│   │       │   ├── preprocess_resnet_50_dataset
│   │       │   │   └── img
│   │       │   └── resnet_50_model
│   │       ├── tf
│   │       │   ├── config
│   │       │   ├── output_ref
│   │       │   ├── preprocess_resnet_50_dataset
│   │       │   │   └── img
│   │       │   └── resnet_50_model
│   │       └── tflite
│   │           ├── config
│   │           ├── mobilenet_v2_model
│   │           └── output_ref
│   └── simulator
│       ├── bin
│       └── lib
├── shufflenet
│   ├── config
│   ├── max_min
│   ├── models
│   ├── preprocess_shufflenet_dataset
│   │   └── img
│   ├── test

接下来我们安装 aipubuilder 也就是第一个工具。

pip install sdk/build/build-tool/AIPUBuilder-3.0.175-cp36-cp36m-linux_x86_64.whl

不出意外顺利安装完毕。 第二个就是 AIPU 仿真器了，SDK 中的这个仿真器有点问题，运行的时候会找不到依赖的 libaipu_simulator_z1.so，我们需要修改一下他的 RPATH，这里我们用 pathelf 这个工具。

sudo apt install patchelf

然后修改 aipu_simulator_z1 的 RPATH。

patchelf --set-rpath "\$ORIGIN/../lib" sdk/simulator/bin/aipu_simulator_z1

然后执行一下，出现下面的内容就说明没问题了

$ sdk/simulator/bin/aipu_simulator_z1
Open File /mnt/home-nas/repo/r329-test/sdk/simulator/bin/runtime.cfg Failed! Exit.
PARSE RUNTIME CONFIG FILE FAILED: /mnt/home-nas/repo/r329-test/sdk/simulator/bin/runtime.cfg
aipu_simulator_z1: src/main.cpp:99: int main(int, char**): Assertion `sim && sim->init(argc, argv) == 0' failed.
[1]    98955 abort (core dumped)  sdk/simulator/bin/aipu_simulator_z1
编译并仿真模型

所有准备工作已完毕，执行下面的命令等待输出结果吧～

cd shufflenet
./compile_and_run.sh

看到下面的信息说明你成功了！

[INFO]:AIPU START RUNNING: BIN[0]
[INFO]:TOTAL TIME: 21.173628s.
[INFO]:SIMULATOR EXIT!
[I] [main.cpp  : 135] Simulator finished.
Total errors: 0,  warnings: 0
class is mixing bowl

我们看到最后输出了分类信息 mixing bowl 它对应的图片其实是这张

虽然它正确的分类应该是 soup bowl，嘛～

上板验证

（占坑，待收到实机后测试）

本文的主要内容就到这里了，后面的内容留给想自己动手编译模型的小伙伴们～

附录
测试原始 ShuffleNet 模型

这里的模型来自于 onnx models 我们可以先用 onnxruntime 测试一下这个模型

$ cd shufflenet
$ python infer_onnx.py
cost time:0.004265546798706055
class for ILSVRC2012_val_00000003.JPEG is Shetland sheepdog, Shetland sheep dog, Shetland
cost time:0.003937959671020508
class for ILSVRC2012_val_00000001.JPEG is Gila monster, Heloderma suspectum
cost time:0.003919124603271484
class for ILSVRC2012_val_00000005.JPEG is crib, cot
cost time:0.0039060115814208984
class for ILSVRC2012_val_00000002.JPEG is ski
cost time:0.0039293766021728516
class for ILSVRC2012_val_00000004.JPEG is soup bowl
生成量化校正集

AIPU 只支持 int8 推理，所以需要提供量化校正集。这里要注意不管你的原始模型输入 layout 是 NCHW 还是 NHWC，生成的校正集通通都得是 NHWC！

cd shufflenet/preprocess_shufflenet_dataset
python preprocess_for_shufflenet_dataset.py

上面的命令就生成了我们所需要的 dataset.npy 这里图片的预处理我们采用了 onnx models 里的处理方法

mean = [0.485, 0.456, 0.406]
var = [0.229, 0.224, 0.225]
生成仿真输入

这里同样不管你的原始模型输入 layout 是 NCHW 还是 NHWC，生成的输入通通都得是 NHWC！

cd shufflenet/test
python gen_input.py

上面的命令就生成了我们所需要的 input.bin 这里的图片预处理我没用校正集那种方法，用了比较简单的方法，经过测试对结果没有太大影响

orig_image = cv2.imread(img_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (input_width, input_height))
image = (image - 127.5) / 1
image = np.expand_dims(image, axis=0)
image = image.astype(np.int8)

image.tofile("input.bin")
编译配置文件

编译模型需要的配置文件如下

[Common]
mode = build

[Parser]
model_type = onnx
input_data_format = NCHW
model_name = shufflenet
detection_postprocess =
model_domain = image_classification
input_model = ./models/shufflenet.onnx
input = gpu_0/data_0
input_shape = [1, 3, 224, 224]
output = gpu_0/softmax_1

[AutoQuantizationTool]
quantize_method = SYMMETRIC
ops_per_channel = DepthwiseConv
reverse_rgb = False
calibration_data = ./preprocess_shufflenet_dataset/dataset.npy
calibration_label = ./preprocess_shufflenet_dataset/label.npy
label_id_offset = 0
preprocess_mode = normalize
quant_precision = int8

[GBuilder]
outputs = aipu.bin
profile= True
target=Z1_0701

我们需要注意的几个配置 - model_type，支持 tensorflow、tflite、onnx、caffe，本文的模型是 onnx - model_domain，指定模型所属的领域，本文的模型用于图片分类，所以是 image_classification，其他的还有 object_detection keyword_spotting image_segmentation 等 - calibration_data 指定我们前文生成的量化校正集 - outputs 是最后要上板的模型可执行文件

最后我们可以执行下面的命令来编译模型，得到 aipu.bin

cd shufflenet
aipubuild config/onnx_shufflenet_build.cfg
仿真配置文件

仿真模型需要的配置文件如下

[Common]
mode = run

[Parser]
model_type = onnx
input_data_format = NCHW
model_name = shufflenet
detection_postprocess =
model_domain = image_classification
input_model = ./models/shufflenet.onnx
input = gpu_0/data_0
input_shape = [1, 3, 224, 224]
output = gpu_0/softmax_1

[AutoQuantizationTool]
quantize_method = SYMMETRIC
ops_per_channel = DepthwiseConv
reverse_rgb = False
calibration_data = ./preprocess_shufflenet_dataset/dataset.npy
calibration_label = ./preprocess_shufflenet_dataset/label.npy
label_id_offset = 0
preprocess_mode = normalize
quant_precision = int8

[GBuilder]
inputs=test/input.bin
simulator=../sdk/simulator/bin/aipu_simulator_z1
outputs=test/output.bin
profile= True
target=Z1_0701

我们需要注意的几个配置 - inputs 指定我们前文生成的仿真输入 - outputs 指定模型仿真输出 - simulator 是我们复制到 sdk/simulator/bin 里的 AIPU 仿真器路径

最后我们可以执行下面的命令来编译模型，得到 test/output.bin

cd shufflenet
aipubuild config/onnx_shufflenet_run.cfg
