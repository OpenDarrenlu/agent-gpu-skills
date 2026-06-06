# 文本风格转换 Paper List

**作者**: 罗福莉做难而正确的事情。请各自媒体不要再神化和消费个人

**原文链接**: https://zhuanlan.zhihu.com/p/39211679

---

Papers of Non-parallel (Unpaired) Data

1. 把style和content 隐式分开，建立起non-style的content representation
Style Transfer from Non-Parallel Text by Cross-Alignment, NIPS-2017, Tianxiao Shen,

[paper], [code]

总结：相当不错的论文，值得再次深入研读！！这篇文章主要是为了建立一个shared latent content space Z。最简单的VAE可能会建模进去全部的信息（我的理解是不能过滤style的信息），所以用了对抗训练的方法。一种是直接在Z上对抗，一种是在各自的decoder端的时候进行professor forcing（teacher forcing和self-feed）的对抗。后者进行了cross-align的过程，所以模型也叫做Cross-aligned auto-encoder training。

Style Transfer in Text: Exploration and Evaluation, Zhenxin Fu from PKU, AAAI-2018

[paper], [code]

总结：这篇文章主要关注点在于如何获取一个不包含style的content representation。类似于多任务学习，用了一种对抗训练的方式提取出content representation，再接着提出两个模型如何基于content representation进行decode。模型一是过两个decoder，模型二是过将style embedding拼接到content vector上，再过同一个decoder。

Style Transfer Through Back-Translation. Shrimai Prabhumoye. ACL-2018

[paper]

总结：这篇文章用了一个很巧妙的方法得到了一个不包含style的content vector。作者先把带style1的句子通过一个预训练好的英翻法的机器翻译，得到style1的法语句子，再通过法翻英系统，把encoder端的隐藏层表示拿出来（命名为z）。作者说由于这两个翻译系统是在non-style的文本上训练的，所以z应该是non-style的content表示，从而再根据content的表示z过两个不同的deocder端，得到不同style的文本。最后，为了确保两个decoder端的确decode出了不同style的文本，还加了一个分类器一块儿训练。用了Gumbel-softmax解决了由于decoder端离散输出作为分类器输入的问题。

2. 把style和content 显式分开，直接删掉带有style的词
Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer, Juncen Li, ACL-2018, NAACL-2018

[paper] [code&data]

一作是微信李俊岑。思路是先删去src句子中的情绪词，留下content，再根据从语料里Retrieve和src的句子content相似但是sentiment相反的句子，并从中抽取出相反情绪的词x‘，再根据content和x’生成句子。文章提出了多种Generate方式，从简单的基于模板到RNN。实验阶段也比较详细，人工为测试集写了答案，并人工判分（算是包含最多的人工打分paper）

Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach, Jingjing Xu from PKU, ACL-2018

[paper],[code]

总结：只做文本情绪风格转换，重点突出在如何“保留content”。不同于其他paper都是重点关注如何建立起一个不包含的style的content表示，这篇文章中心是：先显式得去掉情绪词(跟Juncen Li思想相同但方法不同)，再用不带情绪词的sequence生成目标情绪的文本。由于这两步中间的信息传递是离散的sequence，所以用强化学习去训练。




3. 最新paper： 不做content和style的区分
A Dual Reinforcement Learning Framework for Unsupervised Text Style Transfer. Fuli Luo. IJCAI-2019.

[paper][code]

不做content和style的分离，直接学习两种风格之间的一步映射关系。这是我在JCAI19的一篇工作，但这个motivation在我18年的时候关注这个task就有了。如下图，左边是之前的方法的抽象图，x和y分别表示两种不同风格的句子，先做风格和内容分离，再做融合生成。右图这篇文章的架构，直接学习两种风格之间的一步映射模型，也就是学习两个seq2seq模型。

为什么不做分离更好，我认为有这个原因有两个方面：

一是从是否能得到很好的风格表示角度来说：

显式的风格和内容分离(上面的第二类)，对于风格隐含表达而非在字层面区分而言，那么很难做好分离，因此一般这类的方法比较适合情感转换，因为情感词和非情感词是较为容易区分的。
那么对于隐式的风格和内容分离的（上面的第一类），通常是通过一个风格判别器D去做的。如果一个隐层表示c（通常是一个dense vector）表示能很好的迷惑判别器D，那么之前的工作就认为是做到了很好的内容提取。事实上是不是这样子的呢？最近有一个ICLR相关的文献验证了，通过判别器去做内容表示的提取是非常不靠谱的。详细看论文，但是我觉得可以给一个比较直观的解释就是，就算c能够很好的迷惑D，那么c里面是很有可能包含与D能判别的style正交的某些其他style，这样子其实c并不是学习到了纯粹的内容表示。

二是从训练和测试GAP的角度来说：

下图x，y表示两种不同风格的句子。

可以看到，训练阶段（右边）是通过一个重构回自己来训练的，而测试阶段（左边）是交叉融合生成的。这就导致，训练和测试阶段其实走的路线是不太一致的，如果一旦x_c和x_s分离的不好，那么测试阶段就糟糕了。

讲完了motivation，那难点来了，由于缺乏平行数据，怎么去学习两个seq2seq模型呢？受启发于dual-learning，我们通过设计如下的训练框架：

以informal到formal为例，首先把一个informal的句子通过前向的模型f转换为formal的句子y‘，此时怎么判断y‘生成是否好呢？好的标准就是风格转换成功了，内容也得到了很好的保留。因此，为判断风格是否转换成formal的，可以把y‘丢到一个预先训练好的infomal/formal的二分类模型里面去（就用做informal<->formal transfer的语料库训练的），如果y‘被分类为formal的概率(R_c) 很高，那么就做到很好的风格转换啦。其次，怎么判断y‘做到了很好的内容保留呢？可以把y’通过反向的模型g转换回informal的句子x，如果能很高概率（R_s）重构回x了，那么y'就能得到的很好的内容保留啦！因此通过这两个反馈信号R_c和R_s，就可以通过强化学习来训练模型啦！至于为什么强化学习，因为y‘的输出是离散的tokens，loss的梯度不能精准的回传到前面的网络f。

上面的例子详细讲解了如何训练前向的transfer模型f，那么如何训练后向的transfer模型g呢？如下图：

此外，由于RL依赖于模型要初始化给训练提供warm-start，因此本文采用了一些基于模版的方法来构造一批伪对齐语料来pre-train f和g。此外，为了稳定RL训练，模型也加入了一个teacher-forcing的手段，利用back-translation构造的伪对齐数据通过MLE来训练的。MLE和RL交替训练更新模型。

最后实验结果那自然是相当不错的，大幅超过了近10个baselines。我也开源了code和10个baseline以及我们的模型在测试集合上的生成结果，方便大家复现啦～～ https://github.com/luofuli/DualRL

最后，血的教训：当我把这篇花费大半年心血的paper做出来兴高采烈投了NAACL后，居然因为在涂上方写了个\vspace{-0.4in}而被desk reject（未审稿因为格式原因直接拒）。在此，给大家提个醒，要通过上移图片来压缩文章千万别太明显！

其他paper
Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer. Cicero Nogueira dos Santos, ACL-2018,

[paper]

总结：文本去暴力化。把style1->style2->style1(其中->表示一个encoder-decoder，style1表示暴力文本，style2表示正常文本)。

Unsupervised Text Style Transfer using Language Models as Discriminators, Zichao Yang, Arxiv,

[paper]

总结：把传统的GAN的判别器D从二分类模型改为一个语言模型，给生成器G更多的反馈信息（更好训练得到一个fluent的句子）。 此外，用了Gumbel-softmax解决G、D之间离散的问题。G是encoder-encoder框架，在encoder后走了一个decoder计算重构误差，最后将重构误差和语言模型误差加起来一起优化G。

SHAPED: Shared-Private Encoder-Decoder for Text Style Adaptation, NAACL-2018,

[paper]

总结：文本领域风格迁移，用类似于多任务学习的框架做的（分别用shared encoder来建立起两个领域的公共空间，用两个private encoder建立领域独立的空间。decoder端也类似。）新Idea：复旦大学邱锡鹏的关于GAN用在多任务学习，对抗学出private space和share space，从而保证两个空间不相交。（参考论文Adversarial Multi-task Learning for Text Classification Adversarial Multi-Criteria Learning for Chinese Word Segmentation ）。受到这两篇文章的启发，我们也可以在领域迁移的shared-encoder和private-encoder之间也加上一个对抗学习，保证学到的shared-encoder真的是不包含private domain信息的表示。

Toward Controlled Generation of Text, ICML-2017, [paper], [code]
Unpaired Dataset
Positive<->Negative
情绪风格转换Yelp Review Dataset (Yelp)[Amazon Review Dataset]
Gender transfer把带有性别倾向的句子转化为另外一个性别倾向的句子，
Political slanttransfer不同政治党派说话风格
以上两个都来自：Style Transfer in Text: Exploration and Evaluation, AAAI-2018, [paper], [code&dataset].
Paired (Parallel) Dataset
Formal <->Informal[dataset]: 正规用语和非正规用语的转换，注意是pair的！！而且训练数据很大，有100K [paper]: Dear Sir or Madam, May I introduce the YAFC Corpus: Corpus, Benchmarks and Metrics for Formality Style Transfer, NAACL-HLT 2018
莎士比亚风格<->正常文本风格Shakespearizing Modern Language Using Copy-Enriched Sequence to Sequence Models, EMNLP-2017 Workshop, [paper][code&dataset]
Workshop
Stylistic Variation, EMNLP-2017, [link]
Stylistic Variation, NAACL-HLT-2018, [link]
