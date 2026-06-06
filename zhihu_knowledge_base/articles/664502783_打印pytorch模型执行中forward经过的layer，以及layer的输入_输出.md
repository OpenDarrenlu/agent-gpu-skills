# 打印pytorch模型执行中forward经过的layer，以及layer的输入/输出

**作者**: NULL让智能更廉价: AI芯片软件栈与异构计算系统

**原文链接**: https://zhuanlan.zhihu.com/p/664502783

---

如何详细的看到一个pytorch模型执行过程中forward经过的layer，以及这个layer输入/输出？

----来自GPT4(三次问答)

import torch.nn as nn
import torch

#自定义钩子函数
def print_forward_hook(layer_name):
    def hook(module, input, output):
        print(f"Forward pass through layer: {layer_name} ({module.__class__.__name__})")
        for i, inp in enumerate(input, start=1):
            if isinstance(inp, torch.Tensor):
                print(f"Input {i}: type {type(inp).__name__}, shape {inp.shape}")
            elif isinstance(inp, (list, tuple)):
                print(f"Input {i}: type {type(inp).__name__}")
                for j, item in enumerate(inp, start=1):
                    if isinstance(item, torch.Tensor):
                        print(f"  Item {j}: type {type(item).__name__}, shape {item.shape}")
                    else:
                        print(f"  Item {j}: type {type(item).__name__}")
            else:
                print(f"Input {i}: type {type(inp).__name__}")

        if isinstance(output, (tuple, list)):
            for i, out in enumerate(output, start=1):
                if isinstance(out, torch.Tensor):
                    print(f"Output {i}: type {type(out).__name__}, shape {out.shape}")
                else:
                    print(f"Output {i}: type {type(out).__name__}")
        elif isinstance(output, torch.Tensor):
            print(f"Output: type {type(output).__name__}, shape {output.shape}")
        else:
            print(f"Output: type {type(output).__name__}")
        print()
    return hook
    

#给模型注册钩子函数
for name, layer in model.named_modules():
    if isinstance(layer, nn.Module):
        layer.register_forward_hook(print_forward_hook(name))

#forward时候自动调用hook
outputs = model(samples)




下图是RT-DETR模型forward时候Debug信息:
