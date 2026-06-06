# 搭了一个Claude, Codex, Gemini 协作的 Agent 工作流

**作者**: 笑渐不闻声渐悄​中国科学技术大学  信息与通信工程博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/1977462207890626511

---

最终沉迷于玩 claude code, codex, gemini cli, 心血来潮，搭建了一个三者协作的工作流，分享给大家一起用～

Claude Team
github.com/smart-lty/Claude-Team

其实从8月份就开始高强度用 claude code, 当时还发了一个帖子: 被Gemini和Claude Code狠狠驯服了

第一次接触这些 coding agent 时，claude 的能力还比较有限，只能用来做一些 low-level 的任务，比如简单的复制 / 删除一批文件，写个正则表达式，重构一个简单函数等等，随着 claude opus 4.5, gpt-5.1-codex-max, gemini 3 pro 等一众超强模型横空出世，现在各种 coding agent的能力已经比想象中还要强了。我举一些我自身体验过的例子：

开发 nano-PEARL 的时候，MAT 一直都特别低。当时看到知乎一个哥们做spec，debug半天MAT都上不去 (一段关于SGL Speculative训练框架开发的心路历程) ，结果自己也遇到了这个问题，立马笑不出来。但是！我用 codex，给了一段超详细的 prompt，codex 思考了 20 分钟，准确的帮我找到了 bug！
nano-PEARL 的网页，基本是 claude 一己之力搭建起来的，我只告诉了它，我喜欢 Claude 主页那种风格，于是 claude 10分钟帮我做了一个风格很像的网页 (https://smart-lty.github.io/nano-PEARL/benchmark.html) ；
做目前新的 idea 写论文，codex 读我的代码仓库，用 git 分析我的代码改动，claude 和 gemini 写 abstract 初版，gemini 精读我列举的几篇优秀 paper，反复多轮迭代;
...

举这些例子，就是想说，现在的 coding agent 的能力真的比想象中还要强，我推荐每个读过我博客的人去试用一下 claude code 和 codex。

但最近用的多了，有一些不满足——claude / codex / gemini 仍然有很多不足的地方。比如 claude 喜欢写特别多的文档，写特别多的边界防御性代码，codex 的输出不像人话，交流起来非常费劲，写的 abstract 那叫一个 abstract; gemini 更不用说，某网友被删代码库的经历历历在目。于是最近就在自己捣鼓：怎么让这三个AI agent 一起协作，弥补各自的不足；

以及最重要的一点：省一点 claude code 的 token ！(用过的都知道 claude 的 token 有多贵)

于是经过一番调研，结合自己的使用体验，初步定位 claude 为总控，codex 当工程师，负责具体实现以及debug，gemini当分析师，提供重要的超长上下文分析能力以及补充。具体怎么实现呢？

作为一个 agent 小白，其实实现历程很坎坷：

一开始告诉 claude 怎么运行 codex，可以用 codex exec "prompt" 来执行。但是很快发现，codex 的完整思考过程，包括工具调用，都被塞到了 claude 的 context 里，结果就是，随便调用两次 codex，claude的上下文就爆了；
经过一番搜索，了解到了 sub-agent这个概念。心血来潮，觉得可以把调用 codex 的任务交给一个 sub-agent. 实际用起来还是不行：首先 codex的完整思考和工具调用仍然会被塞到 sub-agent的context里（心疼我的钱包3秒），其次 sub-agent 的执行过程具有很强的不确定性，很多时候它根本不调用 codex，直接自己吭哧吭哧干，然后给我拉一坨大的；
然后尝试了一些 MCP. 市面上其实有很多 codex 的 MCP，但尝试了半天才找到一个可以成功使用的 MCP；
然后就遇到了新的问题，gemini 怎么加进来？通过同样的MCP 方式接入，claude 就开始犯迷糊了：到底该用 codex 还是 gemini？不想了还是自己干吧，然后吭哧吭哧再干我几十刀
最终解决方案：通过AGENTS.md 和 GEMINI.md 严格约束 codex 和 gemini 的行为，并且通过一份详细且严格的 CLAUDE.md 来解决三者协作的问题。

其实到这里，整个框架仍然很 naive，后续还有很多可做的事情：比如 codex也可以作为 mcp 接入 gemini；比如更好的 memory 管理方式 (MCP 一多，context 还是会容易爆，且不要指望 auto-compact 能 work)，比如给这些 agent 更多更强的工具 (MCP); 比如引入 Claude 新发的 code-execution MCP, tool-search tool 等等

感觉 coding agent 真的很好玩！感兴趣的同学也可以找我交流使用体验～

HR 也可以找我聊，明年找工作了 =. =
