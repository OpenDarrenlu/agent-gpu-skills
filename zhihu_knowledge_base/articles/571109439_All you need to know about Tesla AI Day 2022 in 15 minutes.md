# All you need to know about Tesla AI Day 2022 in 15 minutes

**作者**: maja性别男，爱好女

**原文链接**: https://zhuanlan.zhihu.com/p/571109439

---

Tesla AI Day 2022

9月30号东部时间晚上8点，北京时间早上8点，漫长的等待后，本次大会神秘的面纱被缓缓揭开，迎来的马斯克（Elon Musk, 小马哥）不怀好意的笑容。 这可能是召集了Tesla内部最多技术精英的一次盛会，在此之前关于准确的发布日期被跳票了两次。

此前被给予期望，但是举步维艰的机器人"Optimus"，带给人们的惊喜并不大。 “我们不想让它摔倒，但是他能做更复杂的事情”，小马哥欢快地做了下解释，便开始播放预录制视频代替现场展示更加复杂的机器人操作。视频中Optimus在办公室稍显笨拙地进行了一些有限的机械操作：







因此 Optimus更像是一个人形机器人原型，相比于波士顿动力在两足机器人的卓越工作(Boston Dynamics)，更加灵活的结构设计，更加实用的功耗，以及借助Autopilot团队在汽车上的卓越工作成为了Optimus的一大分享亮点。视频中与步履蹒跚的机器人对应是，边角清晰，语义视觉感知。再次证明Tesla在常规AI项目中的优势，这就不得不谈到前AI主管 Andrej Karpathy（小帅）。

时间拉回到3个月前，帅秃的Andrej Karpathy(小帅)在结束长达4个月休假突然宣布离职，一度让外界对于Tesla FSD今后的发展方向产生了质疑。几乎同一时间位于San Mateo的一个数据标注团队（>200）也全部被裁撤（dismissed）。在调查中我们发现，在小帅休假的同时，同样博士毕业于Standford的John Emmons(大帅)，于2022年2月担任高级工程主管接替了小帅在Autopilot视觉团队的工作，与 Mulian,Ashok一同支持AutoPilot团队vision only stack的工作：

Milan

小帅曾经说服了小马哥在Auto Pilot的基础上，基于完全视觉开发 FSD，并在TESLA AI day 进行首次公开的亮相 ：

因此 Andrej Karpathy 之后Tesla FSD, Autopilot vison only stack如何发展就成为大家本次大会首要关注的点。事实上本次亮相的技术咖人数众多，隐晦地展示了Tesla足够的人才储备：




1 Prototype of Optimus Robot Hardware Design (5 mins)
Topic	Subtopic	Speaker
Optimum	intro	小美
	Stuctural design	Malcolm
	Actuator设计，28 -> 6	Konstantinos Laskaris
	hand design	Mike

Elon重申了Tesla对人才的渴望，然后邀请了古铜色皮肤，一头卷发的小美给大家回顾Tesla Bot 整个时间线：

从Concept到 2022年2月的原型机，在最近的一个迭代版本，不得不说Tesla行动远比想象的快， 当然遗憾的是对于车主最为关心，吐槽孱弱的语音能力，甚至被国内语音供应链打得找不着北，仍然没有被重视和更新。将智能从轮子(wheels)复制到机器人，绝非换腿那么简单，小美说Tesla花了很多时间在研究人体结构，人体拥有非常多达28个自由度的关节，可以非常低的功耗去做很多运动。实际上每个自由度对应Optimus这个我称之为原型机的一个传动装置(Actuator)。考虑到制造成本，通过数据学习，28个传动装置被压缩到6种。针对能耗 Tesla 研发可用于机器人全天候工作的电池系统，共包含2.3kWh能量，52V电压电池包，部署在PCB板上的电池管理系统，以及用于支持机器人运行在Tesla SoC上感知，控制，通讯语音等计算能耗：




小美很快邀请Malcolm上台进行更加细致的讲解。Malcolm将汽车碰撞测试引入到结构设计(Structural Design)。这是一个非常具有想象力的工作。Malcolm Burgess正好在Tesla从事车辆动力学研究。Malcolm之前，是前苹果Mac硬件副总裁 Doug负责硬件设计:

Malcolm Burgess（大麦）从2010年加入Tesla，担任Manager已经长达12年之久，可以说是金镀的老tesla了。除了碰撞测试部分, 大麦还介绍了4关节（4 bar link）的非线形受力分析和更高效的力矩设计：


讲完结构部分，大麦就马上邀请已经一头白发的Konstantinos Laskaris (小白) 给大家讲解能耗和传动设计。

不得说传动设计的真不错，在demo中差多1000磅 (~ 450 Kg)的钢琴直接被吊起来做起了引体向上，马力不可谓不足。

小白在特斯兰担任首席电机设计师。电机上电后可以可以通过数据信号线，模拟电流PWM方波通过GPIO输出电压控制信号, 小白曾就职于西门子，从事SPWM的研究，正好是这方面的大拿:


通过数据分析，小白将28个自由度的传动器优化为6个类别，极大的简化了工程落地的难度：







小白认为人类的所有肌肉向一个方向发力时候，也是巨大的，关键是人类灵活的关节和小手，说着说着，使了个颜色就介绍小Mike上来：







小Mike那是更不含糊, 直言“人的手非常复杂(incrediblely/tremendeously dexterous)”：平均每秒可以空间旋转300度，还有不计其数的末端感知，几乎可以抓取我们生活中的任何物体。我们搞了这个拥有11维自由度，包含6个传动器力反馈式仿生人手，就是为了在工厂里学习模仿（Ergonomic）抓取(grasped)那些不仅现在可以看到的，也是未来可能会出现的抓取目标：“the factories and world around us is designed to be ergonomic”.




这样发布会的第一部分，就由Robot的硬件设计组成，而如何让机器人真正工作起来，就需要Tesla软件团队的工作了

2. From Optimus to Autopilot Team (3 mins)




Topic	Subtopic	Speaker
Optimum with Autopilot	AutoPilot software	Milan
	Local motion	小眼镜，大眼镜




首先出场的是瘦高的软件总监Milan Kovac (小高)。小高曾在 Tesla autopilot团队工作达6年之久。承接小Mike, 小高总结了Autopilot的视觉系统在机器人上的应用：




同样采用了稍后即将介绍的占子网络(Spatial Occupancy Network), Optimus唯一的不同可能仅仅在于数据的采集，包括机器人视角和光照环境。令人称赞的是，Autopilot团队还利用了神经辐射场(Radiance fields)技术进行了视觉3D重建工作。

Plenoxels: Radiance Fields without Neural Networks

而由于特斯拉采用占子网络，天然就和稠密建图+光照颜色渲染集大成的神经辐射场技术强相关。

早在2021年Plenoxels证明，无需MLP隐式编码场景，相反他们使用了空间网格(spatial occumancy grid "Spherical Harmonics")。

这一变化可不得了，原来训练RF需要1天时间，现在只需要10分钟，收敛速度大幅提高。后续Nvidia再接再厉利用片上SRAM这个大杀器，将训练时间降低到5s：


这里解释下隐式编码，比如图像的我们用rgb=I(x, y)来表示，那么 F(x, y, rgb) = I(x, y) - rgb = 0 就是隐编码。当F退化为2维度轮廓线时候，就是图像上的闭合曲线，

除了稠密建图（Dense mapping）， Autopilot 团队还探索了视觉导航定位（visual odemetry）。当然我们已经知道了纯视觉导航仅适合室内导航(in-door navigation)场景，这样地理坐标范围较小，空间累积误差小：








得益于仿真，从4月开始，Optimus行走规划控制算法Locomotion在仿真器里飞速的进化，Milan邀请他的同事小眼镜和大眼睛一起讲解Locomotion算法。两足直立行走，对机器来说，从工程角度却比较复杂：Physical Self-Awareness，Energy-Efficient，Balance。

这里提到了Tesal采用了Inverse Kinamics(IK)算法来求解手臂的运动动作，IK算法已经在机械手臂等广泛应用，算是常规操作。这里给大家解释下Inverse Kinamics算法，这是针对多关节非刚体运动，开发的一种运动技术，广泛应用于游戏制作(animation)，工业机器人:




Mulian承诺接下来的6-8个月(2 Qs)的时间内，有可能完成交付并实地部署在工厂里去完成那些机械操作。

3. Tesla FSD (8 mins)




Topic	Subtopic	Speaker
FSD	FSD Arch intro	Milian, Ashok
	Planning	Parel Joint
	Occupancy Network	小黄
	Training Infra	Tim
	Lane Network	John Emmons
	FSD SoC & AI compiler	Sherry
	Auto Labeling	Yangen Zhang
	仿真	David
	数据引擎	Kate Park




FSD Overview

与Mulian一起工作在Autopilot软件团队一起工作的是Ashok Elluswamy登场，是一个印度小哥。后台查了下履历，这小黑哥14年加入tesla，从16奶奶开始担任高级软件工程师（Senior Software Engineer），17年晋升高级主任工程师（Senior Staff Software Engineer），随着Tesla业务扩张于19年开始领导Autopilot 软件团队担任总监，是正儿八经一刀一枪一路升上来十分能打。






在Tesla期间除了负责Auto Labeling，还一直负责视觉场景理解工作。其负责的Fleet learning在2021年CVPR还被前AI主管小帅专门提到过。

这里解释下Fleet learning主要用于迭代/增强同一驾驶路段中模型的表现。最早Tesla主要使用Mobileye的芯片和毫米波雷达(Radar)，在AutoPilot开启时，若存在遮挡或者Radar无法判别的静态物体时，就会上传Tesla数据库。

这和小黑的数据工作契合。这些数据将被用来改善（数据挖掘: 增加特征标签，增广训练数据）模型在这一路况的表现，当其他车辆通过该道路时候，就可以利用这方面的先验信息。




相比于2021年AI day，FSD有了长足的发展，不仅用户迅猛增加，功能也完成了迅速的迭代：35次迭代(月均3次发布)，交付了281个模型。小黑分享了以Occupancy Network + Geometry（lane）/ Object作为训练基础设施（trainning infra）和由自动标注，仿真，数据引擎牵引训练数据的基本架构。

Mulian补充，今年Tesla训练基础设施约有40% ~ 50%的扩张，约有14000 块GPU在数据中心工作。与此同时Tesla开始构建 AI Compiler的工作。实际上由于Dojo的出现，特斯拉就需要适配上层模型算子到自有的底层硬件资源，AI Compiler势在必行。

路径规划

小黑和Mulian的概括性叙述之后，一个叫做Parel Joint工程师紧接着通过一个十字路口行人避让从Planning开始讲解FSD所面临的技术挑战 - 有多种可解的路径，但是必须要选择更安全的方式：

将车速降下来，待行人通过后，迅速通过







不瞒您说，这和驾校主讲精神，真是一模一样。决策树被用于解决这一路径安全偏好选择问题：







Occupancy Network

小黄分享了Occupancy Network。在3d空间（Vector Space）上, 3d物体以volumetric occupancy方式，将通过8个相机捕捉的图像更加精细地表示出来。对于做过3D场景流(scene flow)，以及接触视觉稠密建图（DTAM,Plenoxels）可能就再熟悉不过的数据结构了。

Occupancy Network可以有效地减少对于标签的判别，像边缘石，行人就都可以识别出来。也类似场景流，可以用来做静态障碍物过滤，并预测速度。其延迟(latency)大约在10ms附近，已经是相当快了。事实上绝大多数通过CUDA/Thrust加速的场景流无法达到10ms的速度：











这里FSD的相机输入不是通常的8位通道RGB彩色空间图片，而是每个通道再加4位共12位通道的图片, 多余的4bit用来存储高端相机高动态范围(HDR)的自然光亮度值，曝光度等，更加接近真实光照色彩。


整个模型是一个U形架构, 编码器骨干网络( backbone) 由 RegNet + BiFPN组成特征嵌入(embedding)，并通过多头自注意力机制（multi-headed self attention）技术，进一步编码多相机的特征注意力图(attention)。模仿自然语言的序列编码(position encoding)，匹配到车辆里程计上做长一个编码队列。解码器通过上采样（Deconvolutions实现的）输出FPNd每层对应尺寸的占子网络 (occupancy grid)，路面(surface) ， 场景流 ， 3D语义信息和每个占子网格(voxel)存储的特征(per voxel feature map)。










早在PointNet++，就已经使用了MLP以点云为对象进行场景编码，随后被Nerf借鉴，但是计算速度比较慢。这里Tesla仍然采用MLP进行编码用于特征查询，未来加入自然语言的Beam search topk也不是不可能。

这边Tim 小哥坐不住了，站起来进一步透露整个Occupancy Network需要10000张GPU卡训练一个小时，这可是一笔不小的开支！而剩余的4000GPU卡张用于自动标注工作！

Lane Detection with Language Model




在早期Tesla Auto Pilot产品中，车道线检测问题主要通过图像空间的实例分割解决，包含一个主车道(ego lane)和临近车道(adjacent lanes):

在Autopilot团队最新的系列工作中，从斯坦福获得博士学位的John Emmons（大白）解释道，“产生所有车道，和他们彼此的链接":

看起来很像是端到端的高精道路拓扑地图。高精道路拓扑地图也就是只包含车道线，车道，链接点和其拓扑结构的地图，是高精地图基础图层，也是最重要的数据结构。大白从今年(2022)2月从科学家担任高级工厂经理，接替小帅负责 Auto Pilot 视觉组研发工作：




对于端到端车道线地图网络结构，大白介绍了其包含了视觉模组(vision component)，map component (地图模组)和用于预测车道(Lanes), 链接点(Connector)的语言模型。

其中视觉模组仍然利用了自然语言Transformer， self-attention结构，混合了图像多尺度信息FPN结构, 产生一个视觉特征向量(embedding)与占子网络编码器别无二致（当然大白说convolution+patched image attention一起用的时候，这里可能需要geng xi zhi）。不同的是在Lane network ， 视觉模组和地图模组共同组成编码器。

Map Component 用于预测车道，用于预测拓扑链接点的语言模型，着实吓着很多人，要知道上一次用语言模型来做优化的是谷歌的Tensorflow团队：用于翻译模型去解决模型切分，即Device Placement Optimization (model sharding)问题 -- 这在设备词汇表非常稀缺，且同质的情况下几乎不会有什么有价值的输出。

Map Component的一个作用就是通过道路级地图（就是咱普通车载电子地图）来增强车道的表示，以我对这些从业人员的了解，估计是通过 concatenate操作，拼接不同的特征向量，这个模块叫做Lane Gaidance Module。在预测过程中，多个Attention模块被组合在一个loop里用来预测不同尺度的热力图，用于减少运算：




在18年的时候，就已经有人曾经尝试对高精地图，构建词法分析器，用来解决复杂道路语义问题，并在地图编译上取得了一些非常好的计算效果。Tesla为了能够从视觉表征得到车道分割实例，和以临界矩阵形式存在的道路拓扑，将Conncection, Lane (Lane start, Lane End, Lane Segment)作为Token组成一个自然语言（道路语言）表示（想想看，我们甚至可以给道路语言写出BNF语言描述范式），并不意外。


这样通过Seq2seq方式, 我们将特征向量预测（翻译）出一个道路语言表示，再通过解码器，还原链接电坐标，即从一个domain (visual representation) encode/decode 到另一个domain 道路语言模型，和很像不同语种间的翻译工作。


事实上Tensorflow Brian采用这个思路去做模型切分时，除了Jean Deff，惊讶的发现，不少人来自谷歌翻译团队，当时就纳闷了搞语言翻译的怎么玩起机器学习框架了，读完文章就明白了。

由于编码/解码器的在特征空间的一致性（所表达的意思或逻辑），模型算子和设备的映射就被Seq2Seq方式进行模拟。不过由于众所周知的原因，除了超大规模的异构部署模型外，这个方法并没有很大的实用性。

通常车上视觉模型会检测一些必要的静态障碍物或者地图信息要素，像锥桶检测，停靠车辆等。在统一特征层，多目标模型联合部署前，一直都是单独训练，单独优化。车道线网络，更像是Tesla vision only stack中独具代表性的， 结合了语言翻译模型思想的创新性工作。除此之外，视觉模型就更多地承担动态障碍物检测。

这次大白举了一个十分有趣的，涉及主动安全和防御性驾驶的例子：发现当前车道前方在路口是一辆静止停放车辆，同时红绿灯 亮起，FSD主动变道在另外一辆车辆后方等待红绿灯亮起（当然这个操作很危险，因为在路口变道，过实线，在绝大多数国家属于违章行为）。

FSD SoC + AI Compiler : Inference Latency Optimzation

为了降低视觉模型，尤其是Lane Network (75 million 参数)在 FSD系统的延迟(<= 10 ms)和功耗(8 瓦特)，新上来的印度小哥解释模型特征层产生的都是稀疏的点，尤其是和Attention相关涉及Arg Max, Gather, embedding table look up都是稀疏操作(memory 访问比较分散)，怎么在稠密的矩阵乘法单元进行加速？




首先ViT image block patch 的embedding会被放在一个SRAM，用于高速访存（看起来没少研究Graphcore，在Graphcore早期开源的Bert-base/large MLM/NSP 目标训练解决方案上，如何利用SRAM, 并且大的Embedding放哪里有不少的思考）。

接着通过矩阵乘来实现embedding lookup操作也是语言模型优化的常规手段，很多时候会比Gather/one-hot操作效率高一个数量级。

除此之外，Int8， 片内循环，memory layout，除法加速，基于RDMA的高速通信 -- 几乎从芯片角度可以提升的工作都做了一遍，这可能是一个巨大的工作量，也需要一个强大的技术团队支持：

接着Yanen, David 和 Kate Park分别分享了自动数据标注，仿真和数据引擎。


整套系统恢弘庞大，除了小马哥，我可能不信。早些年为了支持SpaceX推动了一次有争议的“内部交易”，使得SpaceX， Tesla交叉持股。反过来的坐拥全球95%的商业卫星发射订单，以及美国国防部预算的SpaceX和其衍生的星链的成功反而成了Tesla的现金奶牛 -- 这在创业史上，是绝无仅有的。在Tesla依然保持百万年出货量的情况下，小马哥证明过：一切皆有可能。

4 Dojo （5 mins）




Topic	Subtopic	Speaker
Dojo	intro	Pete Bannon, Ganesh V.
	System Update Challenges	Bill Chang
	Actuator设计，28 -> 6	Konstantinos Laskaris
	hand design	Mike










Tesla Dojo芯片架构师Pete Bannon(老白)是一个传奇人物，Pete在加入Tesla之前领导了用于iPhone5的 Arm 32位芯片架构设计，并作为首席架构师推进了用于iPhone5s的Arm 64位芯片设计工作。从履历上，Bannon在16年加入Tesla前，主要在苹果芯片设计部门工作。Dojo可能从那个时候起已经研发了约6年之久。







同Pete一起加入Tesla, 并于2021 Tesla AI Day亮相的是 Ganesh V. (老黑)。老黑在AMD工作14年之久，掌管着约200人的顶尖工程师团队，其中大名鼎鼎的Ryzen芯片就出自老黑和他的团队之手。Dojo项目负责人：




“Many people ask my why Tesla build Super computer” Pete说 “They have some misunderstanding of core nature of Tesla. Tesla is a hardcore technology company” 专注在技术的本质，以及如何造好一辆车。

关于Dojo的design，Pete首次做了公开解释：“We have tried many methods on DRAM， they failed. Then we rejected DRAM and focus on SRAM”。

英国半导体公司Graphcore 的 Mk1 在更早的年份就已经证明了完全在SRAM训练模型的可能性和巨大的收益：无需中断，真正的模型并行（多指令多数据，指令流水线并行）适用于机器学习的低精度计算，以及极低的访存延时。

同时我们看到Dojo新的架构同时采用了多dojo单芯片在集成板上共享HBM的策略。老黑补充为了“我们对数据中心进行了垂直方面的整合”，这让我很快想到了 Graphcore的Mk2000，和NV基于NV Switch扩展机柜解决方案。这使得Dojo 2 与 Dojo 1发生了质的变化：







稍后我们会在 “Rising Star of Semi-Conductor Designer - Tesla” 中通过核心技术参数跟大家做详细的分析，请大家继续关注“聪明的汽车”。


System Updates




Bill Chang已经加入Dojo项目两年，在此之前他主要在蓝色小巨人IBM和苹果工作。为了方便读者，我们先回顾下Dojo1设计。

Recap of Dojo1


Dojo采用了类似Graphcore IPU的分布式SRAM存储。回顾在Dojo1中芯片按如下方式设计：

一个Dojo1共计354个tile(node), 每个Tile包含独立的存储(SRAM)，约1.25 MB, 共计440 MB片上存储，拥有22TFlops FP32和362TFlops FP16浮点计算能力。 这个时候 Dojo还不具备一个cycle发射多个指令的能力，所以采用的SIMD技术。一块dojo1的面积(总计645 mm)还用于部署4*4矩阵乘计算单元（约16x 加速比）和浮点计算器。系统层面25个die（25个dojo1）芯片集成在一起使用：




系统集成层面上，每一个die上都有一个独立的电压控制器，形成完整的dojo1芯片系统集成, 被成为 Training Tile：

Power delivery of Dojo 2 in 2022


Bil提到每个dojo1芯片（die）上计算密度很大，需要达到 1 安培/平方毫米的电流。同时由于Dojo系统垂直方向的系统集成(电压控制器在垂直方向)，整个电能供应也是在垂直方向上集成。在这个方向上还需要考虑热胀冷缩洗漱(Coefficeint of thermal expansion), 以能量损失。通过减少50%的热胀冷缩系数，能量利用提升了接近3倍。


System Tray

System Tray 包含了6个Dojo1集成 (6*25 = 130 dojo1 die) ，这样可以达到54 PFLOPS （> 356 * 6 ~ 2 PFLOPS）。

Dojo Memory Processor


我们惊奇的发现了HBM颗粒，这将是首次大SRAM (25 dies)+ HBM混合计算模型：


这套系统集成了32 GB HBM用于存放数据，达到了片外900 GB/s 数据带宽。在此设计下6块dojo1 training tile 共计 6 * 25 = 150个dojo1 chips 共享20张内寸处理器共计 640 GB HBM，极大缓解了内存紧张问题！Tesla的行动远比我们想象得要快:







和内存处理器集成方向一致，Host机器按垂直方向集成，这样可以给用户提供x86计算环境，以及视频编解码环境：


Dojo机柜解决方案

进一步，通过水平，和垂直拓展形成Tesla Dojo Super Pod:







这样就形成了Tesla Dojo的机柜解决方案。整套下来耗子巨亿万，据场下估计，可能在100亿美金(10 billion)，因此Musk认为未来会以云计算的方式提供租赁服务。

据悉这一整套目前应该期货，在2023年Q1就会有进一步消息。




Dojo AI Compiler

有了这么强大的超级计算机，接下来就是需要软件方面的支持。说起来可能大家不信，Dojo首先从支持Pytorch开始。这是因为因为对于在SRAM上进行计算的芯片，他天然是为静态图设计的。当然如果我们将图拆得足够小，比如一个算子就是一张图的话，实际上就可以完成动态图计算，当然这是以牺牲性能为代价的。

通常针对动态图，会隐藏性地创建静态计算图，先把动态图转化为静态图，然后再将其编译到SRAM上执行计算。由于Pytorch并没有静态编译的概念，是需要厂家先支持静态图编译，然后再支持动转静。在芯片还未成熟之前率先支持Pytorch，收益却是很大，但是难度也就是指数上升了。在这TF2.0正式发布之前，只有Google Brain等团队才有这样的经验。

Rajiv个人经历也比较有趣，在加入Dojo团队之前，就在Waymo，以及更早的Tesla autopilot团队从事自动驾驶系统的研发工作，不少接触自动驾驶PnC和模型方面的工作。


显然想Dojo这样的架构出现，一个核心的问题就是模型切分，和流水线并行。Rajiv小哥展示了 Diffusion 模型在 25 die 的dojo 1 training tile上的效果。模型参数被均衡地分配到25个 dojo chips，只需数分钟就完成了收敛：


这里小哥重点展示了 Reducing 和 Broad casting 操作：

首先我们看到每个 Dojo chip会执行密集的计算，看上去，分布都挺均匀。接着做Local reduction，我们发现很多tile都没有被点亮，这却是符合reduction的计算模型，存在闲置：




本地计算完成后，开始做Global Reduction，我们看到相邻的 chip间进行消息传递。最后会被汇总在中心的chip。

从计算模型上非常契合Graphcore 的 BSP模型，虽然在每个同步阶段设置synchronization barrier。遗憾的是并没有从AI compiler角度分析Parititon的工作。实际上如果考虑算子在Transformed 后的结果，Recompute, Cost Model, partition是由相当的工作量的。


5. 结尾

Tesla AI Day 2022向外界展示了，Tesla强大的人才储备，继续推进vision only stack的工作。视觉方面日臻成熟，取而代之的是Dojo呈现越来越完整的产品规划和对AI compiler越来越强的需求。SpaceX孵化出了星链，Tesla能否从此成为一家芯片设计公司，我们拭目以待
