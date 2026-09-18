# OpenClaw 如何与 Codex/Claude Code/Opencode 丝滑对接：绕过中间商赚差价的实践

**作者**: ZhangZVibe Coding忠实拥趸手撸醋打蒜籽

**原文链接**: https://zhuanlan.zhihu.com/p/2012436123243275090

---

问题的根源

使用 OpenClaw 时，很多开发者都会遇到一个痛点：无法真正与 CLI 编程工具建立直接连接。




ClawHub 上有很多 skill 声称可以连接 Codex、Claude Code、Opencode 等工具，但实际使用时，经常会莫名的卡死。

这是因为大部分的skills使用的指令都会唤醒交互式 TUI 界面，进程会卡住等待用户在 TUI 中输入，OpenClaw 无法正确传递这些交互

即使是使用免交互的cli调用，请求流程也是这样的：

开发者 → OpenClaw → (分析/解释) → Coding Agent → (结果) → OpenClaw → (再处理) → 开发者




这个"中间商"带来几个核心问题：




信息截断：你的所有问题都被 OpenClaw 先分析一遍，可能丢失上下文
开销大：中间openclaw对用户和CLI各转一次手，每次发送消息可能是三倍的价钱
不可控的代理行为：Coding Agent 返回的结果经过 OpenClaw 二次处理，可能包含需要用户选择的内容，OpenClaw 可能自作主张帮你选
交互模式陷阱：很多 skill 直接调用 `opencode run` 这样的命令，但这些都是交互式 TUI 界面，进程会卡住等待用户在 TUI 中输入，而 OpenClaw 无法正确传递这些交互
打破流式输出：因为有中间商需要完整的输出来分析，所有的Coding Agent的输出都是一次性发给用户，用户需要等到天荒地老才能看到二手的输出结果。




第一层优化：非交互式调用




最直观的解决方案是使用非交互式模式：




# ❌ 会卡住（交互式 TUI）
opencode 
# ✅ 非阻塞方式
opencode run -c "你的指令"




加上 `-c` 参数可以在同一个 session 里持续对话，避免每次从头开始丢失上下文。

但这样做只是解决了"卡住"的问题，中间商问题依然存在。

核心突破：绕过 OpenClaw 的信息截断

为了解决根本问题，需要对 OpenClaw 本身进行 hack。我实现的方案是：




1. 免打扰模式开关

增加两个命令：

进入免 OpenClaw 打扰模式
退出免打扰模式




2. 直接通道建立




在免打扰模式下，建立从终端到 Coding Agent 的直接通道，绕过 OpenClaw 的消息拦截和再处理逻辑。

传统方式:
开发者消息 → OpenClaw 解析 → bash pty:true → Coding Agent → 输出 → OpenClaw 处理 → 开发者

绕过后:
开发者 → [免打扰模式] → 直接终端会话 ↔ Coding Agent → 开发者
3. 上下文保持




通过 session 管理，确保与 Codex 或 Claude Code 的对话能保持完整的上下文链。




代码实现




完整实现已开源：https://github.com/deciding/handclaw




核心逻辑：




状态管理：跟踪免打扰模式的激活状态
会话保持：使用 OpenClaw 的 process 工具管理后台会话
消息路由：在免打扰模式下，将用户消息直接转发到 Coding Agent session，跳过 OpenClaw 的自然语言处理
退出机制：提供安全的退出方式，恢复 OpenClaw 的正常功能
plan/build: 可以自由切换plan build模式 （长期切换：!code switch plan, 临时切换-仅当前消息：!plan [message]）




实际效果




启用免打扰模式后，与 Claude Code 的对话流程：




用户：帮我重构这个 API，添加错误处理
↓ (直接转发)
Claude Code: 收到，正在分析代码结构...
↓ (直接返回)
用户：这里的异常处理可以更好
↓ (直接转发)
Claude Code: 同意，建议这样修改...




对比传统方式：

用户：帮我重构这个 API，添加错误处理
↓
OpenClaw: 理解用户想要改进 API...(分析)
↓
Claude Code: 收到...
↓
OpenClaw: Claude Code 完成了任务，它说...(再处理)
Thinking1
Thinking2
Thinking3 (流式输出）
↓
用户：这里的异常处理可以更好
↓
OpenClaw: 用户可能对结果不满意，我来判断一下...(可能误解)
流式输出，直通opencode，可切换plan build
使用建议




什么时候用免打扰模式？




✅ 推荐场景：

多轮迭代的代码重构
需要保持完整上下文的复杂任务
调试和错误修复循环
代码审查和评审




❌ 不推荐场景：

简单的一行命令任务
需要 OpenClaw 协调多工具协作的任务
文件读取/编辑等简单操作




最佳实践




明确边界：在进入免打扰模式前，明确任务范围
定期同步：完成关键里程碑后，退出免打扰模式向 OpenClaw 同步进展
会话管理：及时清理不用的后台 session，避免资源占用。（我的使用习惯是session一个就够了，如果一个项目需要并发任务可以直接git worktree）




总结




OpenClaw 的Coding Agent集成能力很强，但默认的"中间商"模式在某些场景下会成为瓶颈。通过实现免打扰模式，可以：




保持完整上下文：Coding Agent 收到原始输入，不会丢失信息
减少不确定性：避免 OpenClaw 自作主张的代理行为
提升交互体验：与 Coding Agent 的对话更直接、更流畅，有流式输出




这本质上是在"OpenClaw 的协调能力"和"Coding Agent 的原生能力"之间提供一个切换开关，让开发者根据具体任务选择最合适的交互模式。







项目地址：https://github.com/deciding/handclaw

适用版本：OpenClaw v2026.1+

支持的 Coding Agent：Codex、Claude Code、Opencode

如果你有更好的集成方案，欢迎在评论区分享！
