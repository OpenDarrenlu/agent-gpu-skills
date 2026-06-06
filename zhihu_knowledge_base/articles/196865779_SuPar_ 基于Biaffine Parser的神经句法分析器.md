# SuPar: 基于Biaffine Parser的神经句法分析器

**作者**: 张宇月之暗面

**原文链接**: https://zhuanlan.zhihu.com/p/196865779

---

SuPar 是一个以Biaffine Parser (Dozat and Manning, 2017)为基本的架构的Python句法分析工具，提供了一系列的state-of-the-art的神经句法分析（包含依存句法和成分句法）解析器的实现：

Biaffine Dependency Parser (Dozat and Manning, 2017)
CRFNP Dependency Parser (Koo et al., 2007; Ma and Hovy, 2017)
CRF Dependency Parser (Zhang et al., 2020a)
CRF2o Dependency Parser (Zhang et al, 2020a)
CRF Constituency Parser (Zhang et al, 2020b)

项目地址为https://github.com/yzhangcs/parser

上述解析器对应的预训练模型，可以直接加载，用来方便地解析依存句法树或者成分句法树. 此外，这个包中还包含了一系列流行的算法的实现，比如MST (ChuLiu/Edmonds)，Eisner，CKY，MatrixTree以及TreeCRF等等.

原始的Biaffine Parser使用词性Embedding作为模型编码器的辅助输入，这里SuPar还提供了利用CharLSTM或者BERT产生的字表示或者子词表示，来代替词性. 其中CharLSTM是默认的选项，避免了额外的词性标注器来生成词性的麻烦，此外相比BERT更加的高效. SuPar中的 BERT模块是基于transformers中的预训练模型来产生特征. 由于其中的预训练模型互相都是通用的，因此我们的句法分析器也与其他的语言模型，比如XLNet，RoBERTa以及ELECTRA等等兼容，可以方便的切换.

发布的预训练模型中，依存和成分句法上的CRF模型分别是我们近期发表在ACL 2020和IJCAI 2020上的工作，欢迎引用！

@inproceedings{zhang-etal-2020-efficient,
  title     = {Efficient Second-Order {T}ree{CRF} for Neural Dependency Parsing},
  author    = {Zhang, Yu and Li, Zhenghua and Zhang Min},
  booktitle = {Proceedings of ACL},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.acl-main.302},
  pages     = {3295--3305}
}

@inproceedings{zhang-etal-2020-fast,
  title     = {Fast and Accurate Neural {CRF} Constituency Parsing},
  author    = {Zhang, Yu and Zhou, Houquan and Li, Zhenghua},
  booktitle = {Proceedings of IJCAI},
  year      = {2020},
  doi       = {10.24963/ijcai.2020/560},
  url       = {https://doi.org/10.24963/ijcai.2020/560},
  pages     = {4046--4053}
}
安装

SuPar可以方便地通过pip安装：

$ pip install -U supar

或者可以clone仓库在本地安装

$ git clone https://github.com/yzhangcs/parser && cd parser
$ python setup.py install

为了安装成功，下面的依赖需要被满足：

python: 3.7
pytorch: 1.4
transformers: 3.0
性能

目前，SuPar发布了中文和英文的预训练模型. 英语模型在Penn Treebank (PTB)上训练，训练集共39,832句，中文模型在46,572的Penn Chinese Treebank version 7 (CTB7)上训练.

模型的性能和速度都列在下表. 需要注意的是PTB模型的评价需要忽略标点，而中文的CTB7不需要.

所有的结果都是在CPU型号为Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz，以及GPU为Nvidia GeForce GTX 1080 Ti的机器上测试.

用法

SuPar 可以很方便的被调用. 只需几行代码，就可以下载预训练模型，完成句法树的预测：

>>> from supar import Parser
>>> parser = Parser.load('biaffine-dep-en')
>>> dataset = parser.predict([['She', 'enjoys', 'playing', 'tennis', '.']], prob=True, verbose=False)
100%|####################################| 1/1 00:00<00:00, 85.15it/s

调用 parser.predict 将返回一个supar.utils.Dataset的实例，其中包含了预训练的句法树. 对依存句法而言，你可以访问dataset中的每个句子，或者访问所有预测树的某个域.

>>> print(dataset.sentences[0])
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _

>>> print(f"arcs:  {dataset.arcs[0]}\n"
          f"rels:  {dataset.rels[0]}\n"
          f"probs: {dataset.probs[0].gather(1,torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")
arcs:  [2, 0, 2, 3, 2]
rels:  ['nsubj', 'root', 'xcomp', 'dobj', 'punct']
probs: tensor([1.0000, 0.9999, 0.9642, 0.9686, 0.9996])

如果预测时参数prob=True，那么预测树对应的概率也将会被返回. 对于CRF模型而言，如果mbr=True，也就是说使用了MBR（Minimum Bayes Risk）解码，那么模型将返回边缘概率.

注意一下SuPar需要预先tokenize好的句子作为输入. 如果需要解析未tokenize好的原始文本，可以通过调用nltk.word_tokenize来首先做一下tokenization：

>>> import nltk
>>> text = nltk.word_tokenize('She enjoys playing tennis.')
>>> print(parser.predict([text], verbose=False).sentences[0])
100%|####################################| 1/1 00:00<00:00, 74.20it/s
1       She     _       _       _       _       2       nsubj   _       _
2       enjoys  _       _       _       _       0       root    _       _
3       playing _       _       _       _       2       xcomp   _       _
4       tennis  _       _       _       _       3       dobj    _       _
5       .       _       _       _       _       2       punct   _       _

如果有大量的句子需要解析，那么先将他们存储在文件中可能是更好的选择. SuPar同样支持从文件中加载，并且如果指定参数，会将结果保存在pred文件中.

>>> dataset = parser.predict('data/ptb/test.conllx', pred='pred.conllx')
2020-07-25 18:13:50 INFO Load the data
2020-07-25 18:13:52 INFO
Dataset(n_sentences=2416, n_batches=13, n_buckets=8)
2020-07-25 18:13:52 INFO Make predictions on the dataset
100%|####################################| 13/13 00:01<00:00, 10.58it/s
2020-07-25 18:13:53 INFO Save predicted results to pred.conllx
2020-07-25 18:13:54 INFO 0:00:01.335261s elapsed, 1809.38 Sents/s

请确保文件必须是CoNLL-X格式. 可能对原始文本而言，CoNLL-X的某些域的值（比如词性）是缺失的，那么可以预先用下划线填充一下. 同样已经有现成的接口，支持将文本转换为CoNLL-X格式的字符串.

>>> from supar.utils import CoNLL
>>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
1       She     _       _       _       _       _       _       _       _
2       enjoys  _       _       _       _       _       _       _       _
3       playing _       _       _       _       _       _       _       _
4       tennis  _       _       _       _       _       _       _       _
5       .       _       _       _       _       _       _       _       _

对于Universial Dependencies (UD)而言，CoNLL-U格式也是允许的，预测的时候，文件里的一些不规则的行，比如注释，或者行开头的索引不是整数都可以保留下来，且在预测完之后的后处理的过程中可以恢复.

>>> import os
>>> import tempfile
>>> text = '''# text = But I found the location wonderful and the neighbors very kind.
1\tBut\t_\t_\t_\t_\t_\t_\t_\t_
2\tI\t_\t_\t_\t_\t_\t_\t_\t_
3\tfound\t_\t_\t_\t_\t_\t_\t_\t_
4\tthe\t_\t_\t_\t_\t_\t_\t_\t_
5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_
6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_
7\tand\t_\t_\t_\t_\t_\t_\t_\t_
7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_
8\tthe\t_\t_\t_\t_\t_\t_\t_\t_
9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_
10\tvery\t_\t_\t_\t_\t_\t_\t_\t_
11\tkind\t_\t_\t_\t_\t_\t_\t_\t_
12\t.\t_\t_\t_\t_\t_\t_\t_\t_

'''
>>> path = os.path.join(tempfile.mkdtemp(), 'data.conllx')
>>> with open(path, 'w') as f:
...     f.write(text)
...
>>> print(parser.predict(path, verbose=False).sentences[0])
100%|####################################| 1/1 00:00<00:00, 68.60it/s
# text = But I found the location wonderful and the neighbors very kind.
1       But     _       _       _       _       3       cc      _       _
2       I       _       _       _       _       3       nsubj   _       _
3       found   _       _       _       _       0       root    _       _
4       the     _       _       _       _       5       det     _       _
5       location        _       _       _       _       6       nsubj   _       _
6       wonderful       _       _       _       _       3       xcomp   _       _
7       and     _       _       _       _       6       cc      _       _
7.1     found   _       _       _       _       _       _       _       _
8       the     _       _       _       _       9       det     _       _
9       neighbors       _       _       _       _       11      dep     _       _
10      very    _       _       _       _       11      advmod  _       _
11      kind    _       _       _       _       6       conj    _       _
12      .       _       _       _       _       3       punct   _       _

成分句法可以以类似的方式解析. 返回的 dataset对象保存了所有以nltk.Tree格式存储的句法树.

>>> parser = Parser.load('crf-con-en')
>>> dataset = parser.predict([['She', 'enjoys', 'playing', 'tennis', '.']], verbose=False)
100%|####################################| 1/1 00:00<00:00, 75.86it/s
>>> print(f"trees:\n{dataset.trees[0]}")
trees:
(TOP
  (S
    (NP (_ She))
    (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
    (_ .)))
>>> dataset = parser.predict('data/ptb/test.pid', pred='pred.pid')
2020-07-25 18:21:28 INFO Load the data
2020-07-25 18:21:33 INFO
Dataset(n_sentences=2416, n_batches=13, n_buckets=8)
2020-07-25 18:21:33 INFO Make predictions on the dataset
100%|####################################| 13/13 00:02<00:00,  5.30it/s
2020-07-25 18:21:36 INFO Save predicted results to pred.pid
2020-07-25 18:21:36 INFO 0:00:02.455740s elapsed, 983.82 Sents/s

类似于依存句法，一个空句子（仅包含文本）可以方便地先被转换为nltk.Tree格式的空树：

>>> from supar.utils import Tree
>>> print(Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], root='TOP'))
(TOP (_ She) (_ enjoys) (_ playing) (_ tennis) (_ .))
训练

如果要从头训练一个解析器，更加推荐使用命令行，更加灵活，并且参数可定制. 这里是一些训练的例子：

# Biaffine Dependency Parser
# 一些共同和默认的参数保存在了config.ini
$ python -m supar.cmds.biaffine_dependency train -b -d 0  \
    -c config.ini  \
    -p exp/ptb.biaffine.dependency.char/model  \
    -f char
# 如果要使用BERT，需要指定`-f`和`--bert`（默认是bert-base-cased）
# 使用XLNet也是可行的，可以指定`--bert xlnet-base-cased`
$ python -m supar.cmds.biaffine_dependency train -b -d 0  \
    -p exp/ptb.biaffine.dependency.bert/model  \
    -f bert  \
    --bert bert-base-cased

# CRF Dependency Parser
# 对CRF模型而言，需要使用`--proj`选项来丢弃所有的非投影训练实例
# 可选地，可以使用`--mbr`来进行MBR decoding
$ python -m supar.cmds.crf_dependency train -b -d 0  \
    -p exp/ptb.crf.dependency.char/model  \
    -f char  \
    --mbr  \
    --proj

# CRF Constituency Parser
# CRF成分句法分析器的训练过程类似于依存句法
$ python -m supar.cmds.crf_constituency train -b -d 0  \
    -p exp/ptb.crf.constituency.char/model -f char  \
    --mbr

请使用python -m supar.cmds.<parser> train -h命令来得到更多训练方面的提示.

可选地，SuPar在setup.py里注册了一些等价的命令，相比于上面冗长的命令可以稍微简短一点: biaffine-dependency, crfnp-dependency, crf-dependency, crf2o-dependency和crf-constituency.

$ biaffine-dependency train -b -d 0 -c config.ini -p exp/ptb.biaffine.dependency.char/model -f char

这里同样支持分布式训练，来容纳大模型：

$ python -m torch.distributed.launch --nproc_per_node=4 --master_port=10000  \
    -m supar.cmds.biaffine_dependency train -b -d 0,1,2,3  \
    -p exp/ptb.biaffine.dependency.char/model  \
    -f char

更多细节可以在PyTorch的documentation和tutorials中找到.

评价

评价的方式和预测类似：

>>> parser = Parser.load('biaffine-dep-en')
>>> loss, metric = parser.evaluate('data/ptb/test.conllx')
2020-07-25 20:59:17 INFO Load the data
2020-07-25 20:59:19 INFO
Dataset(n_sentences=2416, n_batches=11, n_buckets=8)
2020-07-25 20:59:19 INFO Evaluate the dataset
2020-07-25 20:59:20 INFO loss: 0.2326 - UCM: 61.34% LCM: 50.21% UAS: 96.03% LAS: 94.37%
2020-07-25 20:59:20 INFO 0:00:01.253601s elapsed, 1927.25 Sents/s
参考文献
Timothy Dozat and Christopher D. Manning. 2017. Deep Biaffine Attention for Neural Dependency Parsing.
Terry Koo, Amir Globerson, Xavier Carreras and Michael Collins. 2007. Structured Prediction Models via the Matrix-Tree Theorem.
Xuezhe Ma and Eduard Hovy. 2017. Neural Probabilistic Model for Non-projective MST Parsing.
Yu Zhang, Houquan Zhou and Zhenghua Li. 2020. Fast and Accurate Neural CRF Constituency Parsing.
Yu Zhang, Zhenghua Li and Min Zhang. 2020. Efficient Second-Order TreeCRF for Neural Dependency Parsing.
