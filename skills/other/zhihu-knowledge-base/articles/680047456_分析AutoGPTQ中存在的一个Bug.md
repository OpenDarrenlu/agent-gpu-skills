# 分析AutoGPTQ中存在的一个Bug

**作者**: Lin Zhang​香港科技大学 计算机科学技术博士

**原文链接**: https://zhuanlan.zhihu.com/p/680047456

---

背景

在大模型推理技术中，模型量化可以大幅降低模型内存的开销，同时保持不错的精度，因此受到了广泛的关注。其中，GPTQ[1]算法由于发布早、效果好，几乎成为了Int4权重量化（W4A16）的主流算法。

要体验GPTQ模型非常简单。一方面，我们可以安装AutoGPTQ[2]，它集成了高效的W4A16 Matmul算子（包括但不限于Triton和exllamav2实现），方便用户直接加载量化后的huggingface transfomers模型进行推理。

另一方面，AutoGPTQ也支持我们使用GPTQ算法量化某个模型，对于大多数开源模型，更是有人（例如TheBloke）坚持在huggingface models上传量化后的模型权重，供人直接下载。

然而，一个反常的现象吸引了笔者的关注。

在GPTQ论文里，原作者推荐使用非对称量化方法，因为非对称量化往往能比对称量化取得更好的效果，这一点在其他量化论文中也得到了验证[3]。然而，在TheBloke上传的大量GPTQ模型权重中，却偏偏选择了对称量化方法，其中的主要差别仅仅是group size取值有所不同。

非对称量化 vs 对称量化

为了理解两者的不同，让我们先简单介绍一下非对称量化和对称量化。以Int4量化为例，我们需要将一个group里的所有浮点数映射到[0, 15]的(unsigned) Int4整数。

对于非对称量化，我们需要将其中的最小值xmin映射到0， 最大值xmax映射到15，也即

y = round(x / scale + zeropoint) 

其中，scale = (xmax - xmin) / 15, zeropoint = round(-xmin / scale). 值得注意的是，x和scale是浮点数，而量化后的y和zeropoint是整数。为了确保zeropoint位于[0, 15]，我们使xmin最大截断到0，而xmax最小截断到0。

同理，对称量化需要得到一组浮点数中的绝对值最大值（absmax），然后将 -absmax映射到0，+absmax映射到15，也即

y = round(x / scale + zeropoint) 

其中，scale = 2 * absmax / 15为浮点数, zeropoint = 8为整数。量化后的结果y需要截断到[0, 15]这个范围内[4]。

无论是非对称量化，还是对称量化，在反量化的过程中，我们只需要计算一样的公式：

x = (y - zeropoint) * scale

其中的差别仅仅是y, zeropoint和scale的取值不同。

也许是为了方便对比， GPTQ只提供了一套反量化算子，即我们需要读取y, zeropoint和scale的取值，得到反量化后的x。哪怕对于对称量化而言，存储以及读取zeropoint完全是不必要的操作，因为zeropoint的取值通通为8。

Int4数据pack和unpack

由于PyTorch并不支持Int4数据类型，我们无法直接存储Int4数据类型的y和zeropoint矩阵。于是，GPTQ将相邻的8个int4数据pack成1个int32数据。例如，[0, 1, 2, 3, 4, 5, 6, 7]这八个数据的pack过程如下：

(0 << 0) + (1 << 4) + (2 << 8) + (3 << 12) + (4 << 16) + (5 << 20) + (6 << 24) + (7 << 28)
=  1985229328 = 0b0111,0110,0101,0100,0011,0010,0001,0000

相应的，我们通过 (1985229328 >> (i * 4)) & 15就可以unpack得到其中第i个数值。

此时，对于GPTQ中qlinear的主要参数[5]，假设权重weight的大小为(in, out)，量化后的线性层会产生(in, out)的int4类型的qweight，pack上下相邻的int4后我们得到(in/8, out)的int32的qweight。对于每一个groupsize，我们需要记录scale和zeropoint，也就是(in/groupsize, out)的scales，以及pack左右相邻的int4后得到的(in/groupsize, out/8)的int32数据类型的qzeros。

加减一和数据溢出

了解以上背景以后，我们终于可以尝试着回答最开始的问题。

从公式出发，不难发现非对称量化的精确度是要高于对称量化的。然而，对于GPTQ算法，社区依旧大量采用对称量化的方式，这其实是因为AutoGPTQ中存在的一个bug，影响了非对称量化的实际表现。

在AutoGPTQ的代码仓库中，我们发现int4类型的qzeros数据，在pack之前会进行一个“不必要”的qzeros -= 1的操作[6]，同时在反量化算子的计算过程中，unpack后的qzeros数据会再进行一次qzeros += 1的操作[7]。

尽管以上加减一操作在数学上可以相互抵消，但是在数据pack和unpack过程中可能产生数据溢出的风险。这是因为zeropoint原本是[0, 15]范围内的整数，其中0减1得到的-1会产生数据溢出，导致pack和unpack后得到的结果大为不同。

例如，[0, 1, 2, 3, -1, 5, 6, 7]这8个数字经过上述的pack和unpack操作后，得到的结果是[0, 1, 2, 3, 15, 15, 15, 15]，这不仅破坏了原来-1这个数据，而且这种破坏会影响到它后面的数据。

加减一操作的存在，使得GPTQ非对称量化存在数据溢出的风险，以至于其效果甚至还不如对称量化。注意，对称量化中zeropoint原本的数值是8，加减一并不会导致数据溢出。

数据溢出检测

为了检查GPTQ非对称量化后的权重是否存在数据溢出的问题，我们将量化模型中所有的qzeros进行unpack，然后检测是否有异常值“15”的存在。通过以上操作，我们发现在llama2-7b-chat非对称量化模型中，有两个线性层存在数据溢出的问题。

幸运的是，由于数据溢出问题并不严重，模型仍然可以正常输出。不幸的是，正是模型相对正常的表现，让人容易忽视加减一带来的后果。

笔者自己是在另一个场景下意识到该问题的严重性，即尝试将AWQ权重转换成GPTQ格式，以支持GPTQ类型的算子推理。AWQ和GPTQ的数据pack方式有些不同，在经过unpack和repack操作之后，我们发现AWQ的权重依旧无法使用GPTQ算子正常推理（输出乱码）。

在反复debug之后，笔者终于意识到问题出现在加减一上，由于AWQ的qzeros中存在着大量的0，减一操作导致大量qlinear层（经检测为46层）发生数据溢出，以至于模型完全不可用。

结论

笔者详细讨论了AutoGPTQ中加减一操作带来的问题，然而，由于大量采用加减一操作的量化模型和高性能算子的存在，简单删除加减一操作会带来很大的兼容问题。为此，笔者留意到AutoGPTQ代码库进行过一次类似的尝试，最终还是因为兼容问题放弃了修改方案[8]。

然而，从长远来看，这个几乎成为“feature”的“bug”，还是应该得到大家的重视。因为数据溢出的问题，已经极大限制了GPTQ中非对称量化的使用，同时对统一不同量化算法的底层算子带来极大的挑战，使得社区不得不花费更多精力维护多种量化方案。

也许，一个折衷的方式，是在quantize_config文件中添加一个offset参数，offset为true或者该参数不存在时算子需要考虑加减一操作（以兼容过去的方案），而offset为false时则移除加减一操作。

参考
^GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
^AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
^AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
^GPTQ对称量化有一个小问题，即当x全部大于0时，xmin取值为0，导致scale=xmax/15，这会将xmax映射到23，截断到15会产生较大误差，具体讨论参考：https://github.com/AutoGPTQ/AutoGPTQ/issues/293
^GPTQ还需要g_idx参数记录每行权重对应的group id，用于act-reorder后的量化推理
^减1操作：https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_triton.py#L127
^加1操作：https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/triton_utils/kernels.py#L147
^https://github.com/AutoGPTQ/AutoGPTQ/pull/354
