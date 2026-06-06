# Good,  Bad, and Ugly of ML for Networked Systems

**作者**: Pentium PRO

**原文链接**: https://zhuanlan.zhihu.com/p/128306241

---

前两天看了The Good, the Bad, and the Ugly of ML for Networked Systems。这是和self-driving DB思路很相似的一个talk，不过是应用到了另一个领域。看看不同方向的研究进展总是会给人一些启发。

这个talk邀请了四位研究network system的大佬，谈谈他们对ML的看法。




(1). Balaji Prabhakar, Stanford University

Balaji认为with lots of data, NN and DL can help find good solutions when model assumptions don't hold。比起一些传统方法（例如linear system/markovian models），ML是可以捕捉到一些更具体的信息。

NNs are very good for:

Function approximation: learn and mimic complex functions
Speeding up computations: fast implementations on GPU/TPU/… [但个人感觉ml for sys的很多场景中很难有这个条件…]

Balaji举了几个具体的例子

Self-Programming Network [7'00''] 。作者也用self-driving car做了类比hhhhh。它的大体思路是开发一个模块来对复杂的网络结构进行sense、infer、learn and control。作者举了一个例子是预测每个交换机上会消耗多长时间(from total time between 2 nodes in the network [TX/RX timestamp of packets], determine time spent in each switch)。一个传统做法是用lasso algorithm [8'20'']，但在网络拓扑很复杂的时候运行速度就很慢。用简易的2 layer NN可以在accuracy差不多的情况下得到更快的运行速度[10'20'']。
用wifi signal情况来定位VR眼镜的地理坐标。一个传统方法叫做MUSIC algorithm，但这玩意有60cm的误差 [11'00'']。后来用ML，就只有4cm的误差了。







(2). Bruce MacDowell Maggs, Duke University

Bruce提出了一个概念Data-driven design of network systems [16'30'']：如果我们有个现成的large scale system并且能从中收集很多trace data，这些dataset可以帮助我们evaluate new system design。这个新system可能和旧系统比较类似，或者用完全不同的方法实现了同一件事。

一个例子是End System Multicast，这是一个purely P2P live video delivery system。在evaluation中他们使用了Akamai的log来检测End System Multicast系统的稳定性。

还有个例子是TLS Certificate revocation checking [19'00''-24'00'']。这个例子没怎么听懂…

而对于ML在Network system的应用，作者举了一个例子Context Delivery Network。下面是几个适合用ML解决的例子：

Predicting client-server performance (for one specific client, predict which server could provide best performance) based on past performance, network measurements, network topology, …
Determining client reputation (is this a bot/legitimate client?) based on past behavior
Estimating physical location of client

它们有一个共同特点，就是penalty for false positive is low，也就是说即使预测错了也不会带来很严重的后果（比如拒绝服务）。比如对于算法预测出的suspicious client，我们可以rate-limit 来自它的request，但不要彻底ban it。再比如对于在P2P网络中预测出但suspicious client，可以让它离开P2P网络，只能download from CDN。

网友提问：你觉得more complex model重要呢，还是more dataset重要呢？
回答：More data -> new way to design system. ML is a tool to help us to do that.







(3). David A. Maltz, Microsoft Research – Azure Physical Networking Team

现代网络的规模是很大的，而high availability is the first priority。这就导致find the cause of perceived network problems is hard。

对于这类问题，一股脑想用ML、AI肯定是不行的。作者分别总结了适合Rule-Based method（类似给数据库调优的DBA老专家）和适合ML的场景：

Use ML when: “I don’t understand the underlyingphysics that causes this. However, I could just see outcomes. I know good vs bad, and I want to try and understand the outcome.”
Use Rule-Based when: “I have a good physical model, and I have good understanding of causes clearly.”

作者举了三个例子来说明。

Optimal Layer-1 topology。假如网络中有一段链路出故障了（比如海底光缆断了），我们希望能尽快重新找到别的链路（require new fiber path），避免某个区域断网。这时我们就需要知道其他fiber path的availability（predict what’s time between failure and time to repair distribution for a new fiber path）。这个靠人工诊断是非常困难的。但是用ML就可以estimate using historical experience with “similar” paths。这里ML的作用就是捕获一些人类难以察觉的特征。

2. Optical performance optimization，这个说白了就是调参数（Open Line System enables us to tune parameters for each link to maximize bits per second the link can carry），但是在这个case里用ML并不是个好选择。 1). 对于Network system来说，availability上最重要的，万一ML调错了可能会造成很严重的后果。 2). 人们对这个系统已经有了很详细的了解，用传统的control theory就已经足够好了，ML可能还不如最粗暴的方法。 但是也有些别的system，我们并不知道哪些参数效果最明显（后面网友答疑时Keith提出一个观点：即使一个很简单的只有几个parameter的系统，放在decentralized network situation中也会变得很复杂），这种情况下就可以用ML来帮助我们了。

3. Network availability，这里的问题是检测出gray switch failure。在这个例子中我们有大量的labeled dataset可用，因此用ML完全可以。

作者最后提出了knowledge gap between ML researchers and system researchers。







(4). Keith Winstein, Stanford University

作者总结了ML in Network System的三种paradigm：

Learn then deploy，这也是最传统的模式，人们训练一个model，然后丢到应用场景上deploy。作者认为这个看起来简单，但实际上可能often harder than we expect。作者用Sprout in NSDI2013举了个例子（用ML做congestion control），它是在美国开发的，而在实际deploy之后人们发现，它在美国的移动网络中表现还可以，但放到印度的网络中效果就很不行了。还有个例子是Pensive in SIGCOMM2017（用ML来确定video bit rate），在实际deploy之后的表现也和paper中很不一样。还有个例子是Google用ML去predict flu trends based on historical search engine queries，它确实可以很好的拟合historical data，但实际部署之后ML预测出的效果却很令人惊讶。

另外，有些人并不知道ML是如何work的，而只是把它当做一个black box，看到它在test dataset上效果不错就用了。这里有个例子是用ML做spam filtering（proposed in 2007）。在这个例子中ML学到了一个特征：来自未来的邮件很可能是spam（比如201x年）。这个rule确实取得了low false-positive rate，但是到2010年时这个rule就完全不能用了hhhhhhh。

因此，作者觉得learn-then-deploy is a challenging pattern, and empirically it’s easy to fool ourselves into premature declarations of success。太精辟了……

2. Deploy and learn，意思是先deploy出去，然后在使用的过程中learn online continuously overtime，这样就可以react quickly to real-world changes。作者认为这个方向valuable, but hard because of the nature of networks. Network systems present unique and interesting challenges worthy of research。一个比较好的例子是QUIC in SIGCOMM2017，Google会每天监测算法的表现，并相应调整。但是在network system中想实现learn online并不是很容易，因为会有以下scenario：

Information is distributed：数据是decentralized，很难放到一起ML
Competing agents are adversarial：作者在HotNets 2017的文章指出：it is impossible for a decentralized congestion-control scheme to be globally asymptotically stable (never mind the quality of the outcome!) over a network with “dumb” bottlenecks (e.g. DropTail queues), if it operates by greedily optimizing an objective function whose only input is the fate of its own traffic (when were packets sent, and which arrived and when)。后面网友答疑时Keith进一步指出，Doing ML to optimize worst-case behavior, especially in the presence of adversarial input is an unsolved problem。包括在前面的调参问题中，也可能有两个效果完全相反的参数，这也是adversarial input。
Compute and data are in different places：比如对于edge devices，其实完全可以在edge端进行ML，不需要传到cloud datacenter。

3. Learn from the machines，Teaching machines to learn to design systems。It’s an old-fashioned AI view, but still valuable。

Machine某种程度上还是很智障的。在teach machine的过程中，即使最后并没有真正deploy ML，人类也可以learn so much about the problem。




网友提问：design system to be machine learnable？do we have design principles？
回答：1). 场景不能太复杂（比如上文的decentralized、adversarial agents，还要看ML是否能converge），比如可以考虑某个简单点的subsystem。 2). 还要选好合适的metrics，强行优化ML performance对系统的整体效果提升不一定好。比如如果understandability、stability也能被量化成指标就好了。3). 要有training dataset




网友提问：network中有哪些问题用ML可能会做的更好呢？
回答：scheduler/branch prediction/query planner in DB。在这些场景中cost of misprediction is not so high、ML可以recognize一些pattern。







这个talk其实时间有点早了（2018年9月）。Keith组对此有一个后续工作是NSDI20上的Learning in situ: a randomized experiment in video streaming。




这篇paper主要关注Video Streaming中的Adaptive bitrate algorithm(ABR) 。ABR server会把video分成若干几秒钟的chunk，分别做成不同的码率(size)。ABR可以用来decide the quality level of each video chunk to send，从而在保证视频质量的同时减少stall，optimize user's quality of experience(QoE)。

ABR就有点像上面提到的Competing agents are adversarial这类问题：大的chunk可以提高视频质量，但占用带宽也大，有可能会stall（these goals are conflict with each other）。作者希望开发一个好用的learned ABR algorithm。

作者从以下三个方面介绍了他们的成果：

1. Confidence intervals in video streaming are bigger than expected

作者为了研究现有ABR algorithm的实际性能，搭建了一个video streaming website(http://puffer.stanford.edu)，对每个用户测试不同的ABR scheme。现有的ARB scheme的实验scale太小了 (only a few network nodes, and lasted only a few hours) ，因此putter对于每种scheme都收集了长达两年的数据。作者举例说明了实验scale大小对结果的影响：

如图分别是实验区间为一天、一周、一个月、8个月的数据。Y轴从下到上表示higher video quality，X轴从左到右表示fewer stalls，因此越往右上方表示效果越好。图中展示的是95% confidence intervals。从一天的数据中可以看出，每个scheme的confidence interval都很大，overlap with each other and indistinguishable。因此小规模、短时间的实验根本看不出差别，测出的看似有提升其实很可能只是noise（扎心了吧hhhhhhhhhh）。

最终，在8个月的数据（more than 55000 user IPs, 13 years of video length, nearly 2 years of video per scheme）中，才看出了比较清晰的区别（narrow the confidence intervals downto 20% of the mean value）。下图是对8-month period放大后的版本：

作者在长时间的实验中发现，stall其实发生的很少，所以长时间、大规模的实验才能发现比较可靠的结果，否则就可能会被statistical noise所迷惑。




2. A simple buffer-based ABR algorithm performs better than expected

作者对近年来的3种ABR algorithm进行了实验：

BBA [SIGCOMM '14]：用了一个简单的linear function，select bitrate based on playback buffer
MPC-HM [SIGCOMM '15]：用 harmonic mean来预测throughput。但这篇paper做了两个assumption：1). Throughput can be modeled with HM. 2). Ignore the size of chunk to send，直接assumes transmission time = predicted throughput*chunk size。 但后来作者发现这两个assumption都不成立。实际网络环境中throughput会根据chunk size不同而变化的。
Pensieve [SIGCOMM '17]：这篇用了猛如虎的reinforcement learning来训练end-to-end ABR control。但这篇也做了assumption：it requires network simulators as training environments, and assumes training in simulation could generalize to wild Internet，它需要在simulator中reply throughput traces。但作者认为想simulate real Internet是很难的，所以实验结果也不一定靠谱。

从实验结果也可以看出，这三者的差距并没有那么大。最简单的BBA其实表现还可以了：

所以simpler algorithms that make fewer assumptions perhaps are more general。又扎心了吧…




3. Our way of outperforming existing schemes is learning in situ (in place on the actual deployment environment)

基于前面的工作，作者提出了一个新的算法：Fugu。它把MPC-HM中的throughput predictor换成了transmission time predictor，根据chunk size to send (chunk by chunk real data) 来预测how long it will take for a client to receive a given chunk。另外，作者预测出的是probability distribution，而不是point estimate，作者发现这样效果更好。

在实现TTP（transmission time predictor）时，learning是in situ的。Training data are sampled and fed into TTP as user streams，learning目标是minimize difference (cross entropy loss) between its predictions and the actual transmission times of chunks。

另外，learning in situ不需要有trace and replay、simulator这种操作，因此更能符合真实的网络环境。

下面左图显示Fugu是唯一一个在quality和stall两方面都能outperform BBA的算法。同时对比Pensieve可以看出，training in simulator、replay throughput trace这种方式是不大行的，很难generalize。即使我们想对其加入一些别的指标来trace，如何选择指标也是个难题。

另外，作者还用自己的puffer platform收集的数据重新训练了Pensieve，相当于用了实际网络环境中的数据，也有一点learning in situ的意思了。上面右图中可以看出，这样retrain之后Pensieve也会有所提升，但还是不如Fugu。从中可以总结出两个问题：1). 如何设计出更加能generalize的trace。这还是很困难的。 2). 我们还是很难faithfully simulate the Internet，simulator终究是不靠谱的，那就不如从根源上解决问题。Learning in situ会是个好方法。

最后，作者把puffer上收集到的数据全都开源了，作为一个open research platform for ABR schemes, network and throughput prediction, congestion control。太良心了。




网友提问：learning in situ的过程中，training data的分布是会有改变的嘛？这个会不会影响算法的confidence interval？
回答：作者分别用了两个版本的Fugu：1). Daily retrained on every single days. 2). 长时间的training。其实这两种版本效果都很好。另外实验中展示的是daily retrained版本。因此某种程度上也能说明随着时间变化，data并没有很大的变化。




一些思考：

在Self-Driving DB中做的很多事情其实和上面提到的trace and replay、training then deploy的模式很像。很多evaluation使用了ycsb、TPC之类的工具包。这些模板化的benchmark是否能模拟真实的部署环境呢？这里其实也会有两个问题：1). Trace靠不靠谱？ 2). 如果还不是很容易simulate，能不能learning in situ？




针对第一个问题，本学期storage system课上讲了FAST2020的一篇文章Characterizing, Modeling, and Benchmarking RocksDB Key-Value Workloads at Facebook，或许能某种程度上解决这个问题。作者发现了ycsb只支持有限的几种key-value分布，并不能很准确的模拟实际情况[YCSB-generated workloads ignore key space localities. In YCSB, hot KV-pairs are either randomly allocated across the whole key-space or clustered together. This results in an I/O mismatch between accessed data blocks in storage and the data blocks associated with KV queries.]。而在本文中，我们将线上的实际数据进行trace，然后replay并且analyze。在分析的过程中，我们重点关注热点数据落在哪些kv区间[ The whole key space is partitioned into small key-ranges, and we model the hotness of these small key-ranges. ]，试图发现其中和业务场景相关的一些pattern，然后根据此来设计benchmark [queries are assigned to key-ranges based on the distribution of key-range hotness, and hot keys are allocated closely in each key-range]。之后我们可以用分析结果改进benchmark的设计，进而可以用来调优RocksDB。

在实验中，作者用了三种typical use case作为application：UDB、ZippyDB、UP2X，并针对这三种application进行分析。trace的时长约为24 hours to 14 days traces，而被分析的characteristics包括：

Query composition
Key and value size statistics and distributions
KV-pair hotness and access count distributions
Query per second (QPS)
Hot key distributions in key-space
Key-space and temporal localities

具体的分析结果可以参考FAST'20上的presentation。然而在storage system中还有一个问题：即使前台顺利模拟出了相似的query type composition、KV-pair hotness distribution、value size distribution等等特征，但后台写入文件系统的过程中，仍然无法确定产生的disk block I/O是否也是相似的，而这正是影响RocksDB性能的一个关键因素。

作者开发了一个framework来进行进一步实验：1). 收集前台的trace，然后在同一台机器上replay，收集I/O stat等信息。2). 用ycsb生成尽量相似的workload并进行相同实验，收集benchmarking期间的I/O stat等信息。3). 对比分析结果。实验发现ycsb果然存在一些问题，作者也针对此进行了改进：

作者把新的benchmark和ycsb做了对比，发现效果果然好多了（越靠近红线表示benchmark和实际指标越接近）：

这个工作对ycsb做了很大的改进，但可以看出它还是有一些assumption的：1).整个工作是基于人工分析过的三种workload 2).只针对了disk block I/O。 但现实场景中的storage system和计算机网络有些特征其实是很相似的：1). 我们并不知道workload会是什么样，有些场景下由于涉及隐私信息，trace and replay workload甚至是不被允许的。 2). 影响storage system性能的因素也有很多，例如compaction造成的CPU开销。这样看选择哪些metrics也是个问题。

ML for DB/storage system也是个最近的热点话题，但大量现有工作也是属于Keith所说的learn and deploy这种模式。learning in situ或许也是一个有前景的思考方向吧。
