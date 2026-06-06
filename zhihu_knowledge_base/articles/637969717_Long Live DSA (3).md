# Long Live DSA (3)

**作者**: 王健飞​

**原文链接**: https://zhuanlan.zhihu.com/p/637969717

---

骗子
“We have been a misunderstood and badly mocked org for a long time. We don't get mocked as much now. ”
-- Sam Altman, OpenAI CEO

2015 年我在旷视开发人脸识别系统，推广中，发现客户对我们的第一印象居然是「骗子」。直到我提着两台 ITX 小盒子机器，到客户现场跑私有测试集看到结果，客户才心服口服。后来我才知道，在我们去之前，他们已经被上一代人脸识别技术折磨多年，算法精度远远达不到业务要求。这让我第一次真正切身体会到深度学习的威力。

2015 年还有一件大事，ResNet 横空出世，时至今日仍是 CNN 最主流架构。自此，任何传统 CV 的人脸识别已经不再有意义，而算力和数据，即使成本提高了若干个量级，也不再有人质疑其落地可行性。

2017 年，英伟达推出 Tesla V100 搭载了第一代 TensorCore, 半精度算力 125 TFLOPS，算力竞赛正式启动。2023 年，最新的 H100 半精度稠密算力 800 TFLOPS，是 V100 的 6.4 倍。

算力竞赛
“From now on, the [gross margin] of search is going to drop forever.”
-- Satya Nadella, Microsoft CEO

2023 年的主题是 GPU 产能的争夺。我们不知道 AI 会不会带来人类的生存危机，但显然互联网巨头——不知是欣喜还是恐惧——希望把握自己的命运，「一位接近英伟达的人士称，字节到货和没到货的 A100 与 H800 总计有 10 万块」。国产 AI 芯片公司，也借此获得一丝喘息的机会。

潘多拉魔盒一旦打开就无法关闭，算法工程师们的心态已经悄然发生了变化，「NLP 已死」的声音不绝于耳。甚至于 CV，固然 22B 依然不能「涌现」神奇特性，但多模态已经蔚然成风，典型特点是 title 中带「Everything」，这是之前重监督的视觉领域是不寻常的。

视觉曾经是讲究小模型的。MobileNet 之于 ResNet，MobileViT 之于 ViT，无数工程师们在模型轻量化上前赴后继，原因无他：视觉最主要落地场景是 AIoT，而 AIoT 被成本和功耗双双限制，如果应用不能持续带来更高的业务价值，那么降本是唯一的出路。既然落地应用受限，那么研究百倍计算和数据量的视觉大模型就很难获得支持。

我们知道，ChatGPT 仅仅是 OpenAI 通向 AGI 道路上的一个阶段性产品。国内行业的兴奋点大多不在 AGI 的愿景，而是更实在的 AIGC。NLP 和 CV/CG 的底层任务，生成或者理解都不再是障碍，而这只需要一个通用的基座模型和足够多的算力。换句话说，「大炮打蚊子」将会成为新常态，算力和数据需求必将普遍提升若干数量级。Nadella 的判断不只对 Search 成立，对所有涉及到智能的业务也都将成立。

算力跃迁
"Only performance path left is Domain Specific Architectures (DSAs)."
-- David Patterson

Patterson 的 RISC-V 暂且不管，DSA 是唯一出路已经是不争的事实。我们看 MLPerf Inference Edge 数据，同是 7nm, 英伟达 Orin 面对高通 Cloud AI 100 在 IPS/W 指标上毫无还手之力。我们将运行深度学习模型的 DSA 统称为 NPU 的话，会发现这并不是个例：GPGPU 的功耗大多浪费在数据搬运上，而非核心计算，而优化数据流正是 NPU 的核心价值。我的经验显示，如果软硬件执行到位，NPU 在制程落后一代的情况下，基础的预期是 NPU 可以获得相对 GPGPU 3x-5x 的能效比。

NPU 设计要看端侧。端侧的成本压力是互联网企业难以想象的，DDR 带宽是其最显像的体现。不同于云端推理可以尽量堆叠 DRAM，端侧是能少则少，64-bit 已经算是高端（进一步扩容的封装成本会显著提升），而 LPDDR 技术的发展又远远跟不上算力需求，这也导致通过 fusion 降低 DRAM 压力的技术，如 PyTorch 2.0 的 torch.compile，或者 LLM 的 FlashAttention，在端侧都属于基本操作。类似的，功耗不仅影响散热设计，在端侧更是直接影响产品可行性。如果稍微看的广一些，大家也会发现端侧 GPU 的设计思路和传统 GPU 完全不同，强如 Apple Silicon 也选择 Tile-based deferred rendering，背后的原理是一致的。

NPU 不是一种技术架构，而是一种 DSA 的 Domain 定义。海外市场百花齐放，无论是 Graphcore, Groq 还是 Tenstorrent, Dojo, 都有其独到之处。从市场上，NV/AMD 之外，再无第三家 GPGPU 容身之地；从技术上，DSA 是唯一能在 AI 领域超越 GPGPU 的技术路线。和大众的认知不同，英伟达在 NPU 研究上相当先进，英伟达采用渐进式的 DSA 改造（TMA 是远比 TensorCore 更加 DSA 的技术），仅仅是因为现有 DSA 的玩家们不够努力，使得英伟达可以靠架构之外的巨大优势弥补 GPGPU 的不足。一旦市场上出现有竞争力的 NPU，英伟达无法完成渐进式改造，那么其护城河 CUDA 反将成为绊脚石。注意：相对 GPGPU 指标提升 50% 对 NPU 来说不是竞争力，提升 300% 才是。

国内市场，因为特殊市场需求，端侧 NPU 设计说世界领先并不为过。云端无论 GPGPU 还是 NPU，目前可惜尚未有太出彩的产品。

NPU 设计是一门艺术，入门门槛不高，做好很难。

对 Domain 的把握和预判：芯片的设计与生产周期一般 2 年，后续又至少有 3 到 5 年的生命周期，因此对技术趋势的判断非常重要。在 5 年前，这个问题是预判到 CNN 的收敛点；在 3 年前，这个问题是预料到 Transformer 的爆火；在今年，这个问题则是对 LLM 技术方向的把握。
软件架构：软件优先级是比硬件更靠前的，没有软件方案，不应该开动硬件研发。面对从训练框架到硬件接口的诸多层次，何时解耦，何时 Co-design, 需要强大的工程经验支撑。
硬件实现：不同于 ISP/Codec 等以功能实现为主的 ASIC，NPU 重计算，非常依赖前后端联合优化。即使公司内都有可能面对部门墙的问题，如果后端外包这种 co-design 就几乎不可能。
市场机会：这需要一点点幸运值。

现在，LLM 是 NPU 的绝佳机会，它几乎一口气解决了所有的问题。从推理到训练，NPU 在云端的份额将会在未来 5 年显著提升。天时地利只差人和，是时候给英伟达一点压力了。

One more thing
"We're going to do what I call mortal computation, where the knowledge that the system has learned and the hardware, are inseparable."
-- Geoffrey Hinton

Mortal Computing 在 10 年甚至更久的时间不会成为主流技术。扩展到百年维度，数字电路预期会走到物理极限，届时 Mortal Computing 将会是不得已而为之的选择，真正的硅基生命也将就此诞生。
