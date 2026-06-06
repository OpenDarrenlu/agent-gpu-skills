# Self-Driving Database

**作者**: Pentium PRO

**原文链接**: https://zhuanlan.zhihu.com/p/122598847

---

2020.4.5更新：补充一些最新的研究趋势

今年的ICDE2020专门安排了一个Self-Managing Database Systems

Learned Index（也不一定就是替换index，别的具有经验性规则的模块也行）

https://arxiv.org/pdf/1907.05443.pdf
http://dsg.csail.mit.edu/mlforsystems/ （这里提到了MIT的SageDB，和CMU的Peloton有异曲同工之妙）
http://people.csail.mit.edu/kraska/pub/sigmod19tutorialpart2.pdf

Query Optimizer / Execution

The brittleness and complexity of the optimizer makes it a good candidate to be learned.
Neo: A Learned Query Optimizer
https://15721.courses.cs.cmu.edu/spring2020/papers/21-optimizer3/trummer-sigmod2019.pdf
https://15721.courses.cs.cmu.edu/spring2020/papers/22-costmodels/p307-sun.pdf
Learned Scheduling: https://web.mit.edu/decima/content/sigcomm-2019.pdf

用ML调参数

目前被研究的最多的了...下文中会详细介绍

Workload forecast

https://www.pdl.cmu.edu/PDL-FTP/Database/sigmod18-ma.pdf
这里面可做的坑其实还挺多的，比如冷热分离【根据数据的访问频度（温度）对数据库进行冷热划分（预测），根据预测结果分离不同温度的数据到相应的存储层级】、智能调度与负载均衡【针对该规模线上服务器集群，调度不同虚拟机到物理机上，增加资源利用率； 同时设计高效的集群管理算法优化负载均衡，提高分布式系统效率】。

相关研究中RL被用到的非常多。因此有热心网友开发了RL framework for system research。




后面是去年9月份写的了...

最近一直在做 ML in Database 相关的工作。偶然发现CMU 19spring的15-721课程竟然专门安排了这个专题，不禁欣喜若狂，赶紧去学习了一下。

Andy提出了self-driving database的概念，意思是DB应该像现在研究正火的无人驾驶汽车一样，能自己调节自己的方向盘（knob），查看前方的路况（适应未来的workload），最终能保证乘客安全下车（自主完成workload，并保证较好的performance）。

self-driving DB的功能需要包括以下几点（PPT的23-24页）：

全自动调优（这个过去研究的比较多了）
choose when to apply action（即需要workload forecast，这个目前研究比较少）
Learn from these actions, and refine future decision making process（从action的效果中学习）

另外这些操作还需要do while system is running，也就是调整过程中不能影响前台的业务处理。

课上提供了一篇综述A. Pavlo et al., Self-Driving Database Engineering, in Unpublished Manuscript, 2019，但是只对cmu的学生开放......后来想办法下载来了这篇文章。读后真的感觉受益匪浅，在这里记录一下。不过这节课的视频和PPT其实已经cover了这篇paper的很多核心观点。




1. introduction

self-driving DBMS是个很宏伟的目标，宏伟到现有的DBMS根本无法支撑（也就是不再是OtterTune那种给MYSQL安个外挂的模式了，而是深度集成到DBMS的设计中）。

之所以外挂模式不再可行了，一个很重要的原因就是existing DBMS architectures cannot support major changes without stressing the system further or requiring expensive restarts（这个后面还会提到）。当然还有很多别的原因（比如减少ML model的搜索空间，不要盲目搜索一些bad cfg。还有一些自己去看原文吧...）

因此本文就来从工程的角度解释一个真正的self-driving DBMS应该怎么从头设计。




2. background

2.1 Taxonomy

既然self-driving DBMS是和无人车做类比的，那么它也可以像无人车一样分成几个level。本节就解释了这几个level。

这一段其实意义不大...略过了

2.2 System Overview

self-driving DBMS的运行过程可以抽象成这么一张图：

它一共分为以下三个步骤：

Phase1、Modeling

这一步就是训练模型啦。在一个self-driving DBMS中，需要包括两种模型：

Model 1. forecast models that predict the application’s future workload and database state

这个模型的作用是预测workload未来的变化情况。具体实现时可以uses workload traces and database statistics to generate forecast model。注意这个是application层面的，只和业务端的workload有关，和DBMS具体的配置无关（These models are independent of the DBMS’s configuration since they are determined by the application）。在3.1节还会详细介绍这部分。

Model 2. predict how the DBMS’s internal sub-systems will respond to configuration changes made by actions.

这个模型就和具体DBMS有关啦。需要trains these models from its internal metrics collected by its performance monitors. It then computes how changes in these models affect the target objective function. This is known as the value function in ML algorithms.。其实有点钦定用强化学习的意思...毕竟action都用上了。之前的工作对这种model的研究比较多。最新（也是最接近这里的思路）的一个成果就是SIGMOD19的CDBTune。

顺口提一下，CDBTune的开源代码在放出不久之后就被移除了...还好有热心网友提前fork了。

另外AutoML中对超参数自动调参的工作很多也在用强化学习方法。

Phase2、Planning

这一步我们要use its models to select actions that provide the best reward (i.e., objective function improvement) given the current state of the system（又钦定强化学习了......）。

在planning这一步中这两种模型都要用。首先用Model 1来预测出未来会遇到的workload（estimates the application’s behavior for some finite prediction horizon using its workload forecast models），再用Model 2来找出可能达到最佳效果的action（searches for an action that achieves the best reward without violating human-defined constraints (e.g., hardware budgets, SLOs)）。

Phase3、Deployment

这一步就是把前面的action应用到DBMS上啦。这一步看起来简单，但也存在两个难点：

how to expose useful information about the DBMS. This includes both the application’s workload and metrics about the DBMS’s internals, as well as how to control its behavior.
需要Ideally each action completes quickly and with little impact on performance（说白了就是改参数尽量别重启就行了...）
3. environment observations

俗话说巧妇难为无米之炊。训练model第一步就是要获得足够的数据，包括the history of the application’s workload (Section 3.1) and its internal runtime profiling metrics (Section 3.2)，这个功能很多DBMS都已经有了。但还有个难点是如何做到do not expose information about their underlying hardware so the system can reuse training data across operating environments (Section 3.3)

3.1 Workload History

之前提到过a self-driving DBMS selects actions based on what its forecast models predict the system will need in the future。因为数据库的硬件配置是会变化的，所以仅仅记录CPU utilization、tuples read/written这种metric就不行啦。

这一步就需要记录workload history，就是与硬件（环境）无关的logical operation（eg: 事务、SQL Query...）。比如我们可以记录下workload history of the transactions and queries that it executes（Each entry in the history contains the logical operation invoked (e.g., SQL) along with its execution context），这样就可以predict the arrival rate of queries and then extrapolate their expected resource utilization [Reference，zhihu]，那么在phase2 planning的过程中，就可以精确的模拟出未来workload的变化情况。

但是history记录的太多了也不行...后面介绍了两个降低overhead的方法：sampling和aggregation。具体自己去看原文吧...

3.2 Runtime Metrics

metric也是很重要的！A self-driving DBMS trains models from metrics that estimate the cost/benefit of actions under varying conditions. Metrics also guide the system to propose/prune candidate actions.

后面讲了一些对metric的处理方法，比如 ensure that related metrics always use the same unit of measurement 等等。

另外对于Runtime Metrics，If the DBMS has sub-components that are tunable, then it must expose separate metrics for those components.。就是说如果不同session/component的knob是分开配置的，那么它们的metric也应该分开计算。Andy在这里举了一个反面教材就是RocksDB...... RocksDB中每个cf（Column Family）都有自己的一套knob/metric，这个我们在之前做AutoTiKV的时候也经历过。假设我们给RocksDB内核套上一个处理SQL的壳（比如MyRocks），这样每个table都可以有多个cf。假如我们有一个cf叫default，那么这个cf的metric如下：

但比如我们需要num of reads/writes这两个metric，在这个cf里就没有统计啦。我们就需要去global statistics中获取这两个metric：

但这里的metric就是所有cf加起来的值了，无法区分每个cf自己的情况，也就很难对每个cf各自的metric进行调优了。但是postgresql和sql server就没有这个问题（个人感觉其实这就是工程上的一个小毛病...分开statistics一下就行了）。

3.3 Hardware Capabilities

这一步是说因为啊大家的硬件配置是不一样的，所以我们要区分不同配置下的metric dataset（Including a DBMS’s hardware profile with its metrics），这样区分开之后，后面可以想一些黑科技的方法来reusing data from different hardware deployments。

那么什么黑科技方法可以做到这一点呢？原文也没具体说，但给出了一些思路。比如比如对于一个DBaaS的场景（提供一个部署好的DB给用户），我们可以把硬件配置也作为一个metric，这样可以让model也来自动弹性调整vps的硬件配置（model需要做到if the DBMS wants to migrate the database to a faster machine, then it needs to estimate how much it will affect the objective function before it decides whether to move），而用户是不需要操心这部分的。

3.4 Objective Function

objective function也就是要优化的目标（比如throughput、latency、hardware costs）。

后面提到对于OLAP或OLTP workload，要优化的指标应该是不一样的。balabala

4. action meta-data

How the DBMS implements and exposes methods for controlling and modifying the system's configuration. Reduce the number of bad cfgs we have to consider in our model.

不知道是啥玩意...感觉更多是说一个支持self-driving的DBMS内部应该怎么设计

4.1 Configuration Knobs

这一步讲如何修改DBMS的knob，还有有些knob不能改（比如file path/ip address之类的，只能人工调了），有些knob有范围限制（比如不能超过内存大小），有些knob会有风险（比如sync off关闭会提高throughput，但会增大丢失数据的几率，这个model也不知道）。OtterTune paper里都见过了就不再赘述了...

注意Tuning Deltas这一小节讲了如何调整knob。既然使用了action这一概念，那么就不能像Gaussian Process Regression那样直接硬塞一个值了。每个action应该是类似这样：[block_cache_size: +10MB] 或者 [block_cache_size: -10MB]，即the action increments or decrements the knob by a fixed amount。（个人一点注释：有些像disable_auto_compaction这种binary的knob，它的action只能是调成1或0。这种knob强调Tuning delta就意义不大了）

另外Andy在课上还讲了一点：对于不同数量级的knob，increment的amount应该是不一样的，这样可以加速收敛：

但他的学生们觉得多此一举，用same increment size就行了......也不知道到底可不可行了

4.2 Dependencies

这个和如何自动调优就没啥关系了...主要讲DBMS设计的时候就要尽可能避免出现有相互依赖关系的knobs。OtterTune那时候还搞了个K-means来自动判断MySQL中有依赖关系的knob，但这样就太麻烦了，结果还不一定准......不如从根源上解决问题。

4.3 Deployment History

这部分好像和之前重复了...就是说要记录internal runtime metrics from the DBMS’s sub-systems, as well as a representation of the DBMS’s state at the moment of the deployment.

5. action engineering

这部分介绍如何把算出来的action（knob推荐值）应用到DBMS上。这里其实坑点很多哒......

5.1 No Downtime

意思就是改knob不要影响处理前台业务。如果不可避免要影响，也要estimate出要受影响的程度（比如 DB重启 >> 长时间变慢 > 短时间变慢 ），并且可以把这个因素加入到ML model的cost function中。

这里有个例子就是更改MySQL的log file size。虽然更改它一定是要重启的，但如果log size从大改到小，那么重启后还需要compact原来的log，这个又要花很多时间。但从小改到大就不用compact了，重启后接着就能工作了。

另外Andy还鄙视了一下现有的DBMS调节不灵活。Based on our evaluation of existing DBMSs in Section 6, we believe that this limitation is entirely due to engineering factors and not some fundamental scientific reason. （摊手）

5.2 No Independent Self-Managed Components

目前有些DBMS已经实现了针对某些特定模块进行自动调优的Sub-system（比如Oracle的self-managing memory，RocksDB）。那么在一个大一统的self-driving DBMS中，这些小的sub-system就可以去掉了，不然会影响全局的self-driving tuning process。

换句话说就是：你们都是弟弟......

5.3 Observable Deployment Costs

estimate the cost of deploying each action.

only deploy one action at a time

5.4 Aborted Actions

self-driving DBMS应该可以拒绝一些不合法的action（比如block cache size大于内存上界了之类的）

5.5 Replicated Training

为了try out more “risky” configurations without affecting the production deployment，我们可以在replica上调优，并且让replica实时同步前端的DB Query workload（Organizations deploy every mission-critical database application in a replicated environment for high availability.），调优好了之后再应用到生产环境上。但在实时同步这个地方会出现一些inconsistency（比如replica改配置之后变得比master慢很多，无法同步；以及master如何向replica同步workload；以及master和replica配置可能都不一样）的问题。具体自己看原文吧，反正他也没给啥具体的解决方案......

5.6 Adjustable Deployment Resources

Selecting how much of a resource to let an action use is hard.

6. Existing systems

前面说了这么多，那么现在现有的DBMS们的表现怎么样呢？我们来看看吧

6.1 Environment Observations

......

6.2 Action Meta-Data

......

6.3 Action Engineering

......

7. Related work

这个就先不管了...

8. Conclusion

Achieving full autonomy (i.e., Level 5 from Section 2.1) has two tracts of research:

(1) novel ML approaches for value and policy functions
(2) novel DBMS architectures that are amenable to autonomous control.

补充一些课上提到但paper里没有的观点：

0. miscellaneous

workload trace

分析当前的workload

Algorithm -> several recommendations -> 人工判断哪个比较好

recommendations包括如何建RDBMS的index（1990s的工作）、如何调knob等

2010s work

用ML为云用户分配VPS（DBaaS，不再调individual DB）

eg：ATC'18 selecta

1. Previous work

之前的工作还有以下不足点（标红的是个人觉得我们可以改进的点）：

1. Human Judgement

User has to make final decision. Tool只提供option。需要人工再挑选，看是否有效

比如CDBTune就需要人工点击调优按钮才开始调优

2. Reaction Measures

Only solve previous problems, cannot anticipate upcoming usage trends/issues.

然而仅trace过去的workload不一定对未来情况的适用（eg：双十一、黑五。这种情况下人工DBA会提前准备好的）。所以需要looking to future，预测未来的workload。

之前的工作有用LSTM预测应用程序资源占用。

3. No Transfer Learning

现有的工作（像OtterTune） tunes each DBMS instance in isolation, cannot apply knowledge learned about one DBMS to another.

因为它 only optimize(train and test) on one single instance

2. Oracle self-driving DBMS

在2017年，Oracle曾号称开发了全球第一个self-driving DBMS，包括如下几个feature：

Automatic Patching：不用重启就可以给DBMS安装security updates
Automatic Indexing：
Automatic Recovery：
Automatic Scaling：
Automatic Query Tuning：就是查询优化。SQL server和IBM DB2也有这个功能。ingres也可以

2, 3, 4三项基本是self-driving的精髓。但Oracle目前的方案还是外挂一个tool，来给DBA推荐一些选项，而不是完全的automatic。它不能forecast workload，也不能transfer learning between several DBs。所以Andy觉得它并不是self-driving......

3. 其他

除了调knob之外，There are many places in the DBMS that use human-engineered components to make decisions about the behavior of the system. We can replace DBMS components with ML models trained at runtime.

Optimizer Cost Models
Compression Algorithms
Data Structures
Scheduling Policies （scheduling policy for transactions：use ml model to make decision on per-application basis）
总结

Andy觉得在现有的DBMS的基础上是很难做到真正的L5 self-driving DBMS的。与其修修补补不如彻底推倒重来。

ML model的意义就在于可以帮我们发现一些人工想不到的东西（我们的AutoTiKV也发现了这一点）。在一段时间内 human-engineering 仍然会存在，但ML model会起越来越多的作用。很多人工实现的feature都可能被ML协助甚至取代。
