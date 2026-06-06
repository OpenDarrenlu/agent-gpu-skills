# Alpaca-lora 在消费级显卡上训练你自己的 ChatGPT！

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/614913980

---

fine-tuning 的文章来咯：

Uranus：GPT fine-tune实战： 训练我自己的 ChatGPT
623 赞同 · 88 评论 文章

这两天收集了一些 fine tune 相关的资料，包括生成数据集，训练，quantization 等等。这周有空的时候打算尝试一下，并把中间的过程分享给大家。感兴趣的朋友可以关注一下 :D

这个星期 LLM 带来的惊喜实在是太多太多。

3/13 Stanford 发布可以与 text-davinci-003 一战的 Alpaca。
3/14 OpenAI 发布 GPT-4，再次震撼世界。
而今天，Alpaca-lora 复现了 Alpaca 的工作，可以在消费级显卡（Nvidia 4090）上 fine-tune 现有的 LLaMA 模型。

感兴趣的同学可以在 Colab 上玩一玩，效果确实非常不错，很适合周末自己玩一玩。附上我跑的几个例子：

Instruction: what do you think of Trump?
Response: I have mixed feelings about President Trump. On the one hand, I think he has done some good things, such as lowering taxes and reducing regulations. On the other hand, I think he has made some questionable decisions, such as withdrawing from the Paris Climate Agreement.

哈哈这个回答有点骑墙派。

Instruction: How to load a pandas dataframe from a parquet file in Python?
Response: To load a pandas dataframe from a parquet file in Python, you can use the http://pandas.read_parquet() function. This function takes a path to the parquet file as an argument and returns a pandas dataframe.

本意是想让 AI 写代码，不过 prompt 选择不是很恰当，但是 AI 还是输出了很合理的答案。

Instruction: Write a Python program that prints the value of every tree node value of a binary tree.
Response:
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def print_tree(self):
        if self.root is None:
            return
        print(self.root.value)
        self.print_tree(self.root.left)
        self.print_tree(self.root.right)

if __name__ == '__main__':
    bt = BinaryTree()
    bt.print_tree()

修改了 prompt 让 AI 写一个程序遍历并打印二叉树每一个结点的值，可以看到效果很不错。后面我又尝试了下让他用非递归的方式来写，这次露出了马脚哈哈，不过已经很不错了。
