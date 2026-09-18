# Attention 的交叉路口，Linear 向左，Sparse 向右

**作者**: 刘侃呵呵

**原文链接**: https://zhuanlan.zhihu.com/p/1986883182214263488

---

先上链接，RTP-LLM 也开源模型了：

https://huggingface.co/RTP-LLM/Qwen3-Coder-30B-A3B-Instruct-RTPurbo
huggingface.co/RTP-LLM/Qwen3-Coder-30B-A3B-Instruct-RTPurbo
通义千问3-Coder-30B-A3B-Instruct-RTPurbo
modelscope.cn/models/RTP-LLM/Qwen3-Coder-30B-A3B-Instruct-RTPurbo
https://mp.weixin.qq.com/s/wFAJ6oG1CsKBJiCBE45BsQ
mp.weixin.qq.com/s/wFAJ6oG1CsKBJiCBE45BsQ




不会讲故事的人从不写 Introduction 和 Preliminary，直接切入正题。




Linear 真真切切地复兴了。Infra 哥一边骂骂咧咧地说模型结构增加了工作量，一边又开开心心地庆幸又能多领一天工资。Linear 几乎完全按死了 AFD、Helix（linear state 再分离我不知道你在想些什么），Dist KVCache 大残（内存可能都够，别说磁盘了），Spec Decoding 略有受伤（Attention 又要回到了 memory bound）。但这又不绝对，随着越来越聪明的脑子进入到打灰行业，老方法完全有死灰复燃的希望。




Linear 效率是否比 sparse 更好？是我们一直在思考的问题。从 Infra 哥单纯的视角观察，难道 SWA 不是 Linear 吗？一样的线性复杂度，一样的常数空间复杂度。RazorAttention，DuoAttention 等方法已经揭示了 Full 可以以几乎无损的代价转化成 Full + SWA 形式。而且很凑巧的是，转化后的 Full ：SWA 的比例和现在主流 Linear 模型的 Full ：Linear 交错的比例竟然是惊人的一致（约 1:4）。难道，宇宙的尽头就是 1:4？




Sparse Attention 的工程实践是一个有意思的话题。对于 Prefill 阶段，长文下 Q Head 的剪裁可以得到近线性的加速比；对于 Decode 阶段，Q Head 和 KV Head 之间的配合会略显诡异。而介于 Prefill 和 Decode 中间的 Prefill with KVCache 阶段，推理系统需要综合考虑 KV Cache 传输、并行模式、计算效率等多方面权衡。




AFD 可能半死不活了，但 Long Context 照进 Vibe Coding 的现实生活时，KV-Cache Centric 正在越来越比 Model Centric 重要。我们也正在实践过程中不断优化，后续陆续会有工程相关的总结分享出来，以及开源代码更新到 RTP-LLM 项目中。




Linear 是否有比 Sparse 更好的建模优势，在同样的计算和空间复杂度下？后训练和预训练谁能抢得 efficiency 大旗？目前似乎还没有定论。同样的 Attention 优化，当然可以一招鲜！T2I T2V 等模型上我们也正在进行一些优化工作，目前有一些玩具型成果，可以加速在家玩图







懂得越多，懂得越少。效率优化需要深度的工程算法结合，需要极致的工程建模优化，也需要深刻的算法洞见。同时，我们正也在 Linear 的算法工程结合优化上努力，左右手可能很难分出胜负，欢迎加入一起左右互搏。我们还在招聘，欢迎简历骚扰！
