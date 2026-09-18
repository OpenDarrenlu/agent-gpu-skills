# Flash Attention2 示意图

**作者**: NULL让智能更廉价: AI芯片软件栈与异构计算系统

**原文链接**: https://zhuanlan.zhihu.com/p/645627275

---

Flash Attention2比起Flash Attention1，将Qi的切块放外循环，这样好处：

交换循环顺序，一次性出一个结果块Oi，使得内循环结束后只做一次rescaling就行，减少非matmul的计算；
第二个箭头:左图除L还是在内循环中，右图拎出内循环
同时这样也方便后续改进Thread Block方案: Q切4个warp，共享kv，warp间不用通信；
      Attention输出的独立性和Q相关，不论Q切的多碎(切成单词都可以)：                 
                只要K循环完整，softmax就能完整算出；
                只要V循环完整，那么这个单词就能输出完整新编码；
      因此KV拆分时候，KV之间需要同步Q的信息； 而Q拆分时候，KV不用同步；

Flash Attention2：

Flash Attention2




而Flash Attention1的结果Oi不是一次性出来，而每次外循环都要重新到HBM里面取上次O(i-1)的结果来做rescaling，需要外循环结束之后才能完整出一个块；

Flash Attention1：

Flash Attention1




详解Q切4个warp原因
