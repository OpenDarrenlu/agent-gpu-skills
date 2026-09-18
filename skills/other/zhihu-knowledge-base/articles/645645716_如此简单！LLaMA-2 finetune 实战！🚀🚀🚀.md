# 如此简单！LLaMA-2 finetune 实战！🚀🚀🚀

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/645645716

---

​
目录
收起
第一步：准备训练脚本
第二步：准备数据集
第三步：准备模型
第四步：启动训练
最终效果

LLaMA-2 一经发布，开源 LLM 社区提前过年，热度居高不下。其中一个亮点在于随 LLaMA-2 一同发布的 RLHF 模型 LLaMA-2-chat。

LLaMA-2-chat 几乎是开源界仅有的 RLHF 模型，自然也引起了大家的高度关注。但 LLaMA-2-chat 美中不足的是不具备中文能力。尽管有些时候可以通过 prompt 的方式让 LLaMA-2-chat 讲一些中文，但大多数的情况下，LLaMA-2-chat 会固执地讲英文。

像模像样的中文回答
固执地讲英文

此外，LLaMA-2-chat 还具备了过高的“道德感”，有些时候让我觉得没那么有趣。

父亲亲吻女儿在 LLaMA-2-chat 看来也是违反道德的

好久没做 weekend project 了，那么，让我们来 fine-tune 自己的 LLaMA-2 吧！按照下面的步骤，我们甚至不需要写一行代码，就可以完成 fine-tunning！

第一步：准备训练脚本

很多人不知道的是，LLaMA-2 开源后，Meta 同步开源了 llama-recipes 这个项目，帮助对 fine-tune LLaMA-2 感兴趣的小伙伴更好地 “烹饪” 这个模型。

第二步：准备数据集

这一步我采用了GuanacoDataset。这个数据集有两个特点：

多语言支持。我没有选择一个纯中文的数据集，因为我希望模型不要只会说中文。我希望模型英文指令下还是能够以英文进行回复。
alpaca 格式。这一点可以省去我重新组织数据集的时间。

GuanacoDataset 中包含了若干个文件：

GuanacoDataset 一览

由于我想要模型具备的是遵循指令的能力，于是选择了guanaco_non_chat-utf8.json。不过后来发现模型收敛速度很快，可能 guanaco_non_chat_mini_52K-utf8.json 会是一个更节省时间的选择。

我们把 guanaco_non_chat-utf8.json放到 llama-recipes/ft_datasets 下，并重命名为 alpaca_data.json。训练时我们将训练集指定为 alpaca_dataset，llama-recipes 将会自动找到这个文件。

至此，数据集准备完成 :)

第三步：准备模型

首先，下载 Hugging Face 格式的权重。下载之前记得去 Hugging Face 申请 LLaMA-2 的权重并生成自己的 Access Token。

申请通过后的页面
点击 access tokens

有了模型权重的访问权限后，我们就能快乐地下载模型啦。我这里为了缓存到本地的指定目录，使用了 huggingface_hub.snapshot_download 来下载权重：

In [1]: import huggingface_hub

In [2]: huggingface_hub.snapshot_download(
            "meta-llama/Llama-2-7b-hf",
            local_dir="/path/to/Llama-2-7b-hf",
            token="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"
        )

如果追求极致的零代码，你也可以使用 git 来下载模型 :)

第四步：启动训练

我用了一块 3090Ti 24GB 进行训练。由于 llama-recipes 内置了对 alpaca 格式的处理，训练命令为：

export CUDA_VISIBLE_DEVICES=0

screen -L -Logfile screen.log \
python llama_finetuning.py --use_peft \
  --peft_method lora \
  --quantization \
  --model_name /path/to/Llama-2-7b-hf \
  --output_dir /path/to/lora \
  --dataset alpaca_dataset \
  --batch_size_training 40 \
  --num_epochs 1

与官方提供的命令不同之处在于：

在 screen 中运行并留下日志。
添加 --dataset 让 llama-recipes 能够直接找到并处理我们 alpaca 格式的训练数据。
添加 --batch_size_training 充分利用显存，提高训练速度。
添加 --num_epochs 仅训练一轮。

而后便是 20 小时的漫长等待 :)

Loss 的下降如下，可以看出模型在 14% （1000/6991）的数据后便基本收敛。

Step（共6991）	Loss
1	1.30
100	1.21
500	0.57
1000/6991	0.33
1500/6991	0.35
3000/6991	0.32
最终效果

还记得我们对模型的预期吗？

能够讲中文，但英文能力仍然具备。
能够遵循指令，完成任务。

让我们来看看效果吧！

这里我偷懒使用了 alpaca-lora 中提供的脚本 generate.py 进行推理，我添加了 repeatition penalty 让模型减少重复：

添加 repeatition penalty

启动 web UI：

python generate.py --base_model /path/to/Llama-2-7b-hf --lora_weights /path/to/lora

可以愉快地和我们自己训练出来的 LLaMA-2 玩耍啦！

遵循指令：

能够遵循指令

英文能力还是在线：

英文能力在线

甚至还能讲日文哦：

还能讲日文

不再有过份的道德感：

正常的道德感

总结一下，自己 fine-tune 的 LLaMA-2 效果上肯定远远不如 LLaMA-2-chat。得到的模型往往写不出复杂的中文句式。不过整个过程是很有意思的，最后得到的模型也完美地达到了我们的预期 :)

最后再安利一下我们的模型推理框架 Xinference ：Uranus：Xinference: 在个人电脑上玩转LLaMA-2！
