# 利用GPU加速，在Orange Pi上跑LLMs

**作者**: 冯思远​上海交通大学 计算机科学与技术博士

**原文链接**: https://zhuanlan.zhihu.com/p/650110025

---

TL;DR

本文展示了GPU加速的LLM在嵌入式设备上以合适的速度顺利运行。具体来说，不到700元的Orange Pi 5 8G上，我们通过机器学习编译（MLC）技术，实现了Llama2-7b以2.5 toks/sec的速度运行，RedPajama-3b以5 toks/sec运行。此外，我们能够在不到900元的16GB版本的Orange Pi 5上以1.5 tok/sec的速度运行Llama-2 13b模型。

背景

开放语言模型的进步已经催生了跨问题回答、翻译和创意任务的创新。虽然当前的解决方案需要高端的桌面GPU甚至服务器级别的GPU来实现满意的性能。但为了使LLM日常使用，我们想了解我们如何在廉价的嵌入式设备上部署它们。

许多嵌入式设备配备了移动GPU（例如Mali GPU）可以用来加速LLM的运行速度。在这篇文章中，我们选择了Orange Pi 5，这是一个基于RK3588的开发板，与Raspberry Pi相似，但也配备了更强大的Mali-G610 GPU。这篇文章总结了我们首次尝试利用机器学习编译，并为该设备提供了开箱即用的GPU加速。

面向Mali GPU的机器学习编译

机器学习编译（MLC）是一种新兴技术，它自动编译和优化机器学习工作负载，并将编译后的工作负载部署到广泛的后端。在写作时，基于Apache TVM Unity，MLC支持的平台包括浏览器（WebGPU, WASM）、NVIDIA GPU（CUDA）、AMD GPU（ROCm, Vulkan）、Intel GPU（Vulkan）、iOS和MacBooks（Metal）、Android（OpenCL）以及Mali GPU（本文）。

基于通用机器学习编译实现Mali代码生成

MLC是建立在Apache TVM Unity之上的，这是一个用于在不同硬件和后端上编译机器学习模型的通用软件栈。为了将LLM编译到Mali GPU上，我们复用了所有现有的编译流程，没有进行任何代码优化。更具体地说，我们成功地部署了Llama-2和RedPajama模型，采取了以下步骤：

复用了模型优化步骤，包括量化、融合、布局优化等；
复用了在TVM TensorIR中的定义的通用GPU内核优化空间，并将其重新运用在到Mali GPU；
复用了基于TVM的OpenCL 代码生成后端，并将其重新运用在到Mali GPU；
复用了现有的用户界面，包括Python API、CLI和REST API。
运行方法

本节提供了一个分步运行指南，以便您可以在自己的Orange Pi设备上尝试它。这里我们使用RedPajama-INCITE-Chat-3B-v1-q4f16_1作为运行示例。您可以用Llama-2-7b-chat-hf-q4f16_1或Llama-2-13b-chat-hf-q4f16_1（需要16GB的板）来替换它。

准备工作

请首先按照这里的指示，为RK3588板设置OpenCL驱动程序。然后从源代码克隆MLC-LLM，并下载权重和预构建的库。

# clone mlc-llm from GitHub
git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm
# Download prebuilt weights and libs
git lfs install
mkdir -p dist/prebuilt && cd dist/prebuilt
git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git lib
git clone https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1
cd ../../..
使用CLI

从源代码编译mlc_llm_cli

cd mlc-llm/
# create build directory
mkdir -p build && cd build
# generate build configuration
python3 ../cmake/gen_cmake_config.py
# build `mlc_chat_cli`
cmake .. && cmake --build . --parallel $(nproc) && cd ..

验证是否编译成功

# expected to see `mlc_chat_cli`, `libmlc_llm.so` and `libtvm_runtime.so`
ls -l ./build/
# expected to see help message
./build/mlc_chat_cli --help

使用mlc_llm_cli运行LLM

./build/mlc_chat_cli --local-id RedPajama-INCITE-Chat-3B-v1-q4f16_1 –device mali
CLI 运行截图
使用Python API

编译TVM runtime（无需编译完整TVM编译器）

# clone from GitHub
git clone --recursive https://github.com/mlc-ai/relax.git tvm_unity && cd tvm_unity/
# create build directory
mkdir -p build && cd build
# generate build configuration
cp ../cmake/config.cmake . && echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)\nset(USE_OPENCL ON)" >> config.cmake
# build `mlc_chat_cli`
cmake .. && cmake --build . --target runtime --parallel $(nproc) && cd ../..

设置PYTHONPATH（可按需添加到bashrc或zshrc）

export TVM_HOME=$(pwd)/tvm_unity
export MLC_LLM_HOME=$(pwd)/mlc-llm
export PYTHONPATH=$TVM_HOME/python:$MLC_LLM_HOME/python:${PYTHONPATH}

运行下列Python脚本

from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout
cm = ChatModule(model="RedPajama-INCITE-Chat-3B-v1-q4f16_1")

# Generate a response for a given prompt
output = cm.generate(
   prompt="What is the meaning of life?",
   progress_callback=StreamToStdout(callback_interval=2),
)

# Print prefill and decode performance statistics
print(f"Statistics: {cm.stats()}\n")
鸣谢

Orange Pi上的LLM主要由张昊霖（Haolin Zhang）完成。mali优化的支持来自冯思远，基础支持来自邵俊儒和侯博涵以及其他社区成员。
