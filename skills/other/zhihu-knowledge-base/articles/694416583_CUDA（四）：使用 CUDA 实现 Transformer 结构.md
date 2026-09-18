# CUDA（四）：使用 CUDA 实现 Transformer 结构

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/694416583

---

​
目录
收起
一、关键算子及模块的实现
1.1 基础算子的实现
1.2 核心模块的实现
二、训练过程的实现
2.1 训练前的准备工作
2.2 训练过程
参考资料

用 CUDA 来实现 Transformer 算子和模块的搭建，是早就在计划之内的事情，只是由于时间及精力有限，一直未能完成。幸而 OpenAI 科学家 Andrej Karpathy 开源了 llm.c 项目，很好地完成了这一目标。

那么本篇则站在巨人的肩膀上，以该项目作为学习材料，尝试分析和解读构建的全过程。希望通过这一过程能够抛开长期以来对于框架层的依赖，更好地理解 LLM 的算子模块、训练过程和计算优化的诸多细节。

一、关键算子及模块的实现

算子及模块的实现是一件非常考验基本功，但又稍显枯燥的事，但是也是整体实现中最为基础的部分。为了更加详细说明关键算子及模块的实现细节，且囿于篇幅，因此各个部分单独成篇。

1.1 基础算子的实现
LayerNorm 算子的原理、前向和反向的基本实现以及多种优化过程
softmax 算子的原理（safe, online）、简单实现与优化、以及高阶优化
Cross Entropy 前向及其与 softmax 组合的反向的过程
AdamW 优化器的基本原理剖析及其实现过程
GELU 激活函数与残差连接的原理及实现过程
当然还有最最基础的矩阵乘法的算子实现与优化过程
1.2 核心模块的实现
Transformer 中的 embedding 层与 LM head 层的实现
self attention 的实现、优化、cuDNN 的调用，及反向过程的实现（上、下期）
二、训练过程的实现

经过以上艰辛的过程实现了基础的算子和模块后，便可以开始进行训练的过程了。

2.1 训练前的准备工作

2.1.1 训练数据下载并分词

由于该部分不涉及 GPU 计算，通过 Python 实现

python prepro_tinyshakespeare.py

该部分主要执行 2 个操作:

训练数据下载，源数据来源于这里
进行分词，并进行训练集和验证集的划分

输出如下所示：

Saved 32768 tokens to data/tiny_shakespeare_val.bin
Saved 305260 tokens to data/tiny_shakespeare_train.bin

2.1.2 标准模型转换

该部分的操作，主要包括以下几点：

加载标准的GPT2，并保存为 FP32, BF16 两个版本的二进制文件。其中为了提高矩阵乘法的效率，将词表大小从 50257 pad 到 50304.
将 tokenizer 转成二进制文件
将输入及对应的中间状态存成二进制文件，以便后续 debug

输出如下所示：

Running pytorch 2.3.0+cu121
using device: cuda
wrote gpt2_tokenizer.bin
loading weights from pretrained gpt: gpt2
loading cached tokens in data/tiny_shakespeare_val.bin
padded vocab size from 50257 to 50304
wrote gpt2_124M.bin
padded vocab size from 50257 to 50304
wrote gpt2_124M_bf16.bin
padded vocab size in reference grads from 50257 to 50304
wrote gpt2_124M_debug_state.bin
iteration 1, loss: 5.2700, time: 69.203ms, tok/s: 3699.27
iteration 2, loss: 4.0597, time: 48.847ms, tok/s: 5240.88
iteration 3, loss: 3.3751, time: 52.642ms, tok/s: 4863.03
iteration 4, loss: 2.8007, time: 60.446ms, tok/s: 4235.22
iteration 5, loss: 2.3153, time: 60.468ms, tok/s: 4233.63
iteration 6, loss: 1.8490, time: 57.286ms, tok/s: 4468.84
iteration 7, loss: 1.3946, time: 60.249ms, tok/s: 4249.06
iteration 8, loss: 0.9991, time: 62.658ms, tok/s: 4085.70
iteration 9, loss: 0.6241, time: 62.270ms, tok/s: 4111.10
iteration 10, loss: 0.3765, time: 62.727ms, tok/s: 4081.15
final 9 iters avg: 58.621ms
peak memory consumption: 2392 MiB
<|endoftext|>One year ago today:
This is the first week since we last spoke.
---------------
2.2 训练过程

2.2.1 训练过程的实现

从流程上来说，CUDA 的实现与 torch 的实现完全一致，下面就其部分关键环节加以解读

从 checkpoint 加载初始模型及分词器
    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, load_filename);

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");
dataloader 初始化及用法
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, &multi_gpu_config, train_tokens_filename, B, T);
    dataloader_init(&val_loader, &multi_gpu_config, val_tokens_filename, B, T);

通过以下方法进行数据迭代

    dataloader_next_batch(&train_loader);
最关键的训练过程，仍然是以下几步操作
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T, false);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);

更多的细节请参考这里，编译及运行如下：

make train_gpt2cu
./train_gpt2cu

部分日志如下：

Multi-GPU support is disabled. Using a single GPU.
+-----------------------+----------------------------------------------------+
| Parameter             | Value                                              |
+-----------------------+----------------------------------------------------+
| input dataset prefix  | data/tiny_shakespeare                              |
| output log file       | NULL                                               |
| batch size B          | 4                                                  |
| sequence length T     | 1024                                               |
| learning rate         | 3.000000e-04                                       |
| max_steps             | -1                                                 |
| val_loss_every        | 20                                                 |
| val_max_batches       | 20                                                 |
| sample_every          | 20                                                 |
| genT                  | 64                                                 |
| overfit_single_batch  | 0                                                  |
| use_master_weights    | enabled                                            |
+-----------------------+----------------------------------------------------+
| device                | NVIDIA GeForce RTX 4090                            |
| precision             | BF16                                               |
+-----------------------+----------------------------------------------------+
| load_filename         | gpt2_124M_bf16.bin                                 |
| max_sequence_length T | 1024                                               |
| vocab_size V          | 50257                                              |
| padded_vocab_size Vp  | 50304                                              |
| num_layers L          | 12                                                 |
| num_heads NH          | 12                                                 |
| channels C            | 768                                                |
| num_parameters        | 124475904                                          |
+-----------------------+----------------------------------------------------+
| train_num_batches     | 74                                                 |
| val_num_batches       | 20                                                 |
+-----------------------+----------------------------------------------------+
| num_processes         | 1                                                  |
+-----------------------+----------------------------------------------------+
num_parameters: 124475904 ==> bytes: 248951808
allocated 237 MiB for model parameters
allocated 2853 MiB for activations
val loss 4.506145
allocated 237 MiB for parameter gradients
allocated 126 MiB for activation gradients
allocated 474 MiB for AdamW optimizer state m
allocated 474 MiB for AdamW optimizer state v
allocated 474 MiB for master copy of params
step    1/74: train loss 4.364526 (acc 4.364526) (43.898880 ms, 93305.343750 tok/s)
step    2/74: train loss 4.501646 (acc 4.501646) (41.996288 ms, 97532.406250 tok/s)
step    3/74: train loss 4.414096 (acc 4.414096) (41.903103 ms, 97643.632812 tok/s)
step    4/74: train loss 3.957428 (acc 3.957428) (41.950207 ms, 97642.203125 tok/s)
step    5/74: train loss 3.606924 (acc 3.606924) (41.790462 ms, 97742.101562 tok/s)
step    6/74: train loss 3.782941 (acc 3.782941) (41.980831 ms, 97703.687500 tok/s)
step    7/74: train loss 3.566690 (acc 3.566690) (41.839615 ms, 97740.304688 tok/s)
step    8/74: train loss 3.718479 (acc 3.718479) (41.866241 ms, 97756.054688 tok/s)
step    9/74: train loss 3.325452 (acc 3.325452) (41.840641 ms, 97776.726562 tok/s)
step   10/74: train loss 3.443181 (acc 3.443181) (41.834496 ms, 97794.687500 tok/s)
step   11/74: train loss 3.848045 (acc 3.848045) (41.981953 ms, 97766.156250 tok/s)
step   12/74: train loss 3.475283 (acc 3.475283) (41.886719 ms, 97768.632812 tok/s)
step   13/74: train loss 3.636771 (acc 3.636771) (41.883617 ms, 97771.476562 tok/s)
step   14/74: train loss 3.253751 (acc 3.253751) (41.863167 ms, 97778.781250 tok/s)
step   15/74: train loss 3.688257 (acc 3.688257) (42.019840 ms, 97749.390625 tok/s)
step   16/74: train loss 3.868455 (acc 3.868455) (42.007553 ms, 97726.750000 tok/s)
step   17/74: train loss 3.870852 (acc 3.870852) (41.896961 ms, 97730.031250 tok/s)
step   18/74: train loss 3.932542 (acc 3.932542) (42.017792 ms, 97708.765625 tok/s)
step   19/74: train loss 3.657176 (acc 3.657176) (41.962498 ms, 97700.648438 tok/s)
step   20/74: train loss 3.743915 (acc 3.743915) (42.046463 ms, 97677.789062 tok/s)
val loss 3.702019
generating:
---
Nay, thou amongst I: he never came to hear this, that I shall not entertain the will of myself.
Call to play, Gregg!
Take W.L<|endoftext|>Your pleas about my ignorance, Oh, move your way, charge, plunder: plan for
This bloody Staff expedition:
---










2.2.2 使用 cuDNN 模块

为了获得更好的性能，接下来使用 cuDNN 模块，编译及运行命令如下：

make train_gpt2cu USE_CUDNN=1
./train_gpt2cu

部分训练日志如下，单卡性能提升约30%：

Multi-GPU support is disabled. Using a single GPU.
+-----------------------+----------------------------------------------------+
| Parameter             | Value                                              |
+-----------------------+----------------------------------------------------+
| input dataset prefix  | data/tiny_shakespeare                              |
| output log file       | NULL                                               |
| batch size B          | 4                                                  |
| sequence length T     | 1024                                               |
| learning rate         | 3.000000e-04                                       |
| max_steps             | -1                                                 |
| val_loss_every        | 20                                                 |
| val_max_batches       | 20                                                 |
| sample_every          | 20                                                 |
| genT                  | 64                                                 |
| overfit_single_batch  | 0                                                  |
| use_master_weights    | enabled                                            |
+-----------------------+----------------------------------------------------+
| device                | NVIDIA GeForce RTX 4090                            |
| precision             | BF16                                               |
+-----------------------+----------------------------------------------------+
| load_filename         | gpt2_124M_bf16.bin                                 |
| max_sequence_length T | 1024                                               |
| vocab_size V          | 50257                                              |
| padded_vocab_size Vp  | 50304                                              |
| num_layers L          | 12                                                 |
| num_heads NH          | 12                                                 |
| channels C            | 768                                                |
| num_parameters        | 124475904                                          |
+-----------------------+----------------------------------------------------+
| train_num_batches     | 74                                                 |
| val_num_batches       | 20                                                 |
+-----------------------+----------------------------------------------------+
| num_processes         | 1                                                  |
+-----------------------+----------------------------------------------------+
num_parameters: 124475904 ==> bytes: 248951808
allocated 237 MiB for model parameters
allocated 1703 MiB for activations
val loss 4.505090
allocated 237 MiB for parameter gradients
allocated 30 MiB for activation gradients
allocated 474 MiB for AdamW optimizer state m
allocated 474 MiB for AdamW optimizer state v
allocated 474 MiB for master copy of params
step    1/74: train loss 4.370480 (acc 4.370480) (362.220551 ms, 11308.027344 tok/s)
step    2/74: train loss 4.505543 (acc 4.505543) (32.261120 ms, 126963.945312 tok/s)
step    3/74: train loss 4.421638 (acc 4.421638) (32.068607 ms, 127354.804688 tok/s)
step    4/74: train loss 3.957438 (acc 3.957438) (32.161793 ms, 127355.234375 tok/s)
step    5/74: train loss 3.605322 (acc 3.605322) (32.076801 ms, 127446.421875 tok/s)
step    6/74: train loss 3.783947 (acc 3.783947) (32.148479 ms, 127438.093750 tok/s)
step    7/74: train loss 3.568082 (acc 3.568082) (32.101376 ms, 127467.859375 tok/s)
step    8/74: train loss 3.717861 (acc 3.717861) (32.174145 ms, 127441.226562 tok/s)
step    9/74: train loss 3.328022 (acc 3.328022) (32.130047 ms, 127447.265625 tok/s)
step   10/74: train loss 3.443542 (acc 3.443542) (32.140289 ms, 127446.437500 tok/s)
step   11/74: train loss 3.848481 (acc 3.848481) (32.272385 ms, 127380.804688 tok/s)
step   12/74: train loss 3.475115 (acc 3.475115) (32.197632 ms, 127361.500000 tok/s)
step   13/74: train loss 3.643850 (acc 3.643850) (32.207870 ms, 127341.093750 tok/s)
step   14/74: train loss 3.253154 (acc 3.253154) (32.180225 ms, 127335.132812 tok/s)
step   15/74: train loss 3.687203 (acc 3.687203) (32.312321 ms, 127279.265625 tok/s)
step   16/74: train loss 3.870205 (acc 3.870205) (32.337921 ms, 127221.812500 tok/s)
step   17/74: train loss 3.872357 (acc 3.872357) (32.256001 ms, 127200.562500 tok/s)
step   18/74: train loss 3.935513 (acc 3.935513) (32.318462 ms, 127160.890625 tok/s)
step   19/74: train loss 3.659346 (acc 3.659346) (32.240639 ms, 127151.242188 tok/s)
step   20/74: train loss 3.746799 (acc 3.746799) (32.408577 ms, 127089.820312 tok/s)
val loss 3.702071
generating:
---
Nay, thoumyself is well reined in the shoulders of raiment by doasons.
His strength, when she drew it from the folds,
was not a hand's comfort with myself.

<|endoftext|>Floral Everstone:
Wherearruck we hear be, for words well uttered;
---

也可以调整参数进行训练，需要更多的数据，命令如下：

python prepro_tinystories.py
./train_gpt2cu -i data/TinyStories -v 250 -s 250 -g 144 -o stories.log -b 32

部分日志如下：

Multi-GPU support is disabled. Using a single GPU.
+-----------------------+----------------------------------------------------+
| Parameter             | Value                                              |
+-----------------------+----------------------------------------------------+
| input dataset prefix  | data/TinyStories                                   |
| output log file       | stories.log                                        |
| batch size B          | 32                                                 |
| sequence length T     | 1024                                               |
| learning rate         | 3.000000e-04                                       |
| max_steps             | -1                                                 |
| val_loss_every        | 250                                                |
| val_max_batches       | 20                                                 |
| sample_every          | 250                                                |
| genT                  | 144                                                |
| overfit_single_batch  | 0                                                  |
| use_master_weights    | enabled                                            |
+-----------------------+----------------------------------------------------+
| device                | NVIDIA GeForce RTX 4090                            |
| precision             | BF16                                               |
+-----------------------+----------------------------------------------------+
| load_filename         | gpt2_124M_bf16.bin                                 |
| max_sequence_length T | 1024                                               |
| vocab_size V          | 50257                                              |
| padded_vocab_size Vp  | 50304                                              |
| num_layers L          | 12                                                 |
| num_heads NH          | 12                                                 |
| channels C            | 768                                                |
| num_parameters        | 124475904                                          |
+-----------------------+----------------------------------------------------+
| train_num_batches     | 28248                                              |
| val_num_batches       | 20                                                 |
+-----------------------+----------------------------------------------------+
| num_processes         | 1                                                  |
+-----------------------+----------------------------------------------------+
num_parameters: 124475904 ==> bytes: 248951808
allocated 237 MiB for model parameters
allocated 13629 MiB for activations
val loss 2.388124
allocated 237 MiB for parameter gradients
allocated 240 MiB for activation gradients
allocated 474 MiB for AdamW optimizer state m
allocated 474 MiB for AdamW optimizer state v
allocated 474 MiB for master copy of params
step    1/28248: train loss 2.386457 (acc 2.386457) (780.998657 ms, 41956.539062 tok/s)
step    2/28248: train loss 3.283106 (acc 3.283106) (215.723007 ms, 151898.468750 tok/s)
step    3/28248: train loss 2.365078 (acc 2.365078) (219.082748 ms, 150703.875000 tok/s)
step    4/28248: train loss 2.213147 (acc 2.213147) (216.836090 ms, 150849.312500 tok/s)
step    5/28248: train loss 2.213779 (acc 2.213779) (219.773956 ms, 150377.437500 tok/s)
step    6/28248: train loss 2.130578 (acc 2.130578) (217.807877 ms, 150392.234375 tok/s)
step    7/28248: train loss 2.087678 (acc 2.087678) (218.108932 ms, 150362.921875 tok/s)
step    8/28248: train loss 2.067904 (acc 2.067904) (217.784317 ms, 150379.125000 tok/s)
step    9/28248: train loss 1.998698 (acc 1.998698) (218.774521 ms, 150290.078125 tok/s)
step   10/28248: train loss 1.996884 (acc 1.996884) (218.299332 ms, 150265.140625 tok/s)
step   11/28248: train loss 1.945235 (acc 1.945235) (219.539459 ms, 150139.640625 tok/s)
step   12/28248: train loss 1.957886 (acc 1.957886) (219.796478 ms, 150017.156250 tok/s)
step   13/28248: train loss 1.908059 (acc 1.908059) (221.998077 ms, 149754.750000 tok/s)
step   14/28248: train loss 1.890713 (acc 1.890713) (221.164474 ms, 149591.015625 tok/s)
step   15/28248: train loss 1.888435 (acc 1.888435) (221.688828 ms, 149417.265625 tok/s)
step   16/28248: train loss 1.852849 (acc 1.852849) (220.255234 ms, 149357.234375 tok/s)
step   17/28248: train loss 1.843927 (acc 1.843927) (221.689850 ms, 149219.046875 tok/s)
step   18/28248: train loss 1.846019 (acc 1.846019) (218.317825 ms, 149294.171875 tok/s)
step   19/28248: train loss 1.843135 (acc 1.843135) (222.004227 ms, 149153.703125 tok/s)
step   20/28248: train loss 1.869874 (acc 1.869874) (220.104706 ms, 149131.281250 tok/s)

该部分对于软件的版本要求比较高，下面是以上采用的版本情况：

OS: Ubuntu 22.04
Driver: 550.54.15      
CUDA: 12.4
PyTorch: 2.4.0.dev20240513+cu121
cuDNN: 8.9.7.29
cudnn-frontend: 1.4.0




参考资料

[1] https://github.com/karpathy/llm.c

[2] GitHub - ifromeast/cuda_learning: learning how CUDA works

浅水池塘莲叶香，红尘道路柳阴长。 —— 彭汝砺 《和初夏》
