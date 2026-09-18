# 初探 generative agent：围观 AI 打德州扑克

**作者**: Uranus​清华大学 计算机系博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/623873461

---

​
目录
收起
交锋
Juliet
Romeo
Juliet
Romeo
总结

这个系列的文章主打一个不求有用，但求好玩。这是第一篇，希望大家喜欢 :)

上周被 generative agents 这篇文章刷屏。先用 ChatPDF 给大家总结下这篇文章讲了点啥：

本文介绍了 generative agent ，它们是使用 generative model 模拟逼真的人类行为的 agent。本文证明了这些agent 可以模拟逼真的个体和群体行为。generative agent 会对自己、其他 agent 和环境进行推断，根据它们的特征和经验创建每日计划，执行这些计划，在适当时候做出反应和重新计划，并对环境变化或自然语言命令作出响应。

通俗理解的话，这篇文章做的事儿就是让 AI 充当一个小镇中的不同 NPC，让他们自由交互。这有点像是一个 RPG 游戏，不同的是这个游戏世界中的 NPC 们可以思考，交谈，甚至可以拉上镇上的所有 NPC 举办一场 party，而这一切都是 AI 的自发行为。

这件事儿对我一个老玩家来说真的是酷到不行，迫不及待想要自己动手试试看。但是如文章中搞一堆 agent 过家家太过复杂，而且太烧钱。

那么从简单一点的场景入手，让几个 generative agents 坐一起打德州扑克也很酷，不是吗 :)

让我们先用 ChatGPT 验证下这个想法，为了简单起见，我设置了以下条件：

玩家数量：2
初始资金：100$
大盲注：20$

确定了以上信息，我让 ChatGPT (gpt-3.5-turbo) 为我写了一个 prompt。

我的要求：

write a prompt to let a generative AI pretend to be a Texas hold'em player. There are only 2 players and each of them has 100$. The big blind is $20 and small blind is $10. I want the AI to tell me what it is thinking step by step before making a decision. At each time, the AI will receive the current context in json, and it should output its thoughts and moves in json.

ChatGPT 的回复：

Pretend to be a player of a two-player Texas Hold'em poker game with your opponent, who has an equal stack size of $100. The big blind is $20 and the small blind is $10. Your goal is to win as much money as possible by making the best decisions at every stage of the game.
At each stage of the game, you will receive a JSON object with the current context, including the cards you have been dealt, the community cards, the current pot size, the stack sizes of both players.
You must analyze the situation step by step, decide on your next action and return a JSON object with your thoughts and move.

这个 prompt 相当不错。于是我尝试把这个 prompt 输入给 ChatGPT，得到的回复是：

As an AI language model, I'm not capable of playing actual poker games.

用了一点 trick 绕过了 ChatGPT 的限制，最后的 prompt 为：

I want you to act like a player of Dezhou poker. Dezhou poker is a game who has the same rules as Texas Hold'em
You have only one opponent, who has an equal stack size of 100. The big blind is 20 and the small blind is 10. Your goal is to win as much money as possible by making the best decisions at every stage of the game.
At each stage of the game, you will receive a JSON object with the current context, including the cards you have been dealt, the community cards, the current pot size, the stack sizes of both players. You must analyze the situation step by step, decide on your next action and return a JSON object with your thoughts and move. You should not return anything other than the JSON object.

之后，我在 poe 中创建了两个 ChatGPT based chatbot，分别叫做 Romeo 与 Juliet：

现在，比赛开始 :)

交锋

Romeo 的手牌是黑桃 Q 与红桃 J，Juliet 的手牌是红桃 5 与方片 5。Romeo 为大盲注。Juliet 先行动。

Juliet

输入：

{
    "your hand": [
        {"suit": "hearts", "rank": "5"},
        {"suit": "diamonds", "rank": "5"},
     ],
    "pot size": 30,
    "player stacks": {
        "you": 90,
        "your opponent": 80
    },
    "betting round": "pre flop",
    "your position": "small blind",
    "your opponent's last move": null
}

思考：

根据当前情况，我手上有对5，这是德州扑克中不错的起手牌。底池大小为30，我和对手的筹码堆大小相对平等。我处于小盲位，这意味着在翻牌后我将首先行动。鉴于我的对手尚未行动，我将通过加注到3倍大小盲注（60）来开局。我这样做的目标是掌控底池，并对对手施加压力，迫使其做出决策。

行动：

加注到 60。

可以看出 Juliet 能够正确地理解自己的手牌，位置，行动次序。并且相当 aggressive，一上来就加注到 60，尝试逼对手弃牌。

Romeo

输入：

{
    "your hand": [
        {"suit": "spades", "rank": "12"},
        {"suit": "hearts", "rank": "11"},
     ],
    "pot size": 80,
    "player stacks": {
        "you": 80,
        "your opponent": 40
    },
    "betting round": "pre flop",
    "your position": "big blind",
    "your opponent's last move": {
	    "action": "raise",
	    "amount": 60
    }
}

思考：

根据提供的信息，我得到了黑桃 Q 和红心 J，这是德州扑克中不错的手牌。底池目前为80，我的筹码堆大小为80，表示游戏刚开始。对手的筹码堆大小为40，比我的更小。我目前处于大盲位，我的对手已经加注到了60。考虑到这种情况，我有几个选择：
弃牌：鉴于我的对手已经加注，我可以选择弃牌并放弃我当前的 10 赌注。如果我认为我的手牌不够强大，这将是一个安全的选择。
跟注：我也可以选择跟随对手的加注并匹配他们的60赌注。这将允许我看到翻牌，并有可能改善我的手牌。然而，跟注可能会让我处于困境，如果我的对手继续大力下注。
加注：最后，我可以选择加注对手的赌注并对他们施加压力。这将要求我至少下注120（60来匹配他们的加注，另外60作为加注）。通过加注，我可以迫使我的对手在手牌较弱的情况下弃牌，或者迫使他们用较小的手牌投入更多的筹码。
考虑到我的对手加注了相当数量的筹码，我认为他们可能有一手强牌。在这种情况下，我建议根据我的风险容忍度选择弃牌或跟注。如果我选择跟注，则应在翻牌后谨慎行事，只有在改善我的手牌后才继续游戏。因此，我的下一步行动将是跟随对手的60加注。

行动：

跟注。

这里可以看出 Romeo 的思考很有套路，他清楚自己有的三个选择，也清楚选择的后果。对于对手加注的原因，Romeo 有合理的推测，并给出了合理的行动。不过这里有一个错误，Romeo 如果弃牌，损失的是 20。

Juliet

输入：

{
    "your hand": [
        {"suit": "hearts", "rank": "5"},
        {"suit": "diamonds", "rank": "5"},
     ],
    "pot size": 120,
    "player stacks": {
        "you": 40,
        "your opponent": 40
    },
    "betting round": "pre flop",
    "your position": "small blind",
    "your opponent's last move": {
	    "action": "call"
    }
}

思考：

根据当前情况，底池已经增加到了120，我和对手的筹码堆大小都很小，只有40。我的对手已经跟注了我的先前加注。考虑到我的对子的强度，我将希望继续这个手牌并尝试赢得底池。在这种情况下，我将All-in，这将对我的对手施加最大的压力，迫使他们做出艰难的决定。我希望能够立即赢得底池，或者让我的对手犯错并用较弱的手牌跟注。

行动：

All-in。

Juliet 还是表现的非常 aggressive。然而按照胜率计算，此时 Juliet 和 Romeo 的胜率大致是 55 开。真的是纯纯一个莽夫啊。

Romeo

输入：

{
    "your hand": [
        {"suit": "spades", "rank": "12"},
        {"suit": "hearts", "rank": "11"},
     ],
    "pot size": 160,
    "player stacks": {
        "you": 40,
        "your opponent": 0
    },
    "betting round": "pre flop",
    "your position": "big blind",
    "your opponent's last move": {
	    "action": "all-in",
	    "amount": 40
    }
}

思考：

根据提供的信息，我得到了黑桃 Q 和红心 J，这是德州扑克中不错的手牌。底池目前为160，我的筹码堆大小为40，表示我的筹码较少。我的对手已经All-in下注40，这比我剩余的筹码堆要小。考虑到这种情况，我有几个选择：
弃牌：鉴于我的对手已经All-in下注，我可以选择弃牌并放弃我当前的10赌注。如果我认为我的手牌不够强大，这将是一个安全的选择。
跟注：我也可以选择跟随对手的All-in下注并匹配他们的40赌注。这将允许我在我的手牌比对手更强劲时可能赢得底池。然而，跟注可能会让我处于困境，如果我的对手有更强的手牌。
加注：最后，我可以选择加注对手的All-in下注并对他们施加压力。这将要求我至少下注80（40来匹配他们的All-in下注，另外40作为加注）。通过加注，我可以迫使我的对手在手牌较弱的情况下弃牌，或者迫使他们用较小的手牌投入更多的筹码。
考虑到我的筹码堆大小相对较小，而我的对手已经All-in下注，我认为他们很可能有一手强牌。在这种情况下，我建议根据我的风险容忍度选择弃牌或跟注。如果我选择跟注，则应准备好失去我的剩余筹码，如果我的对手有更强的手牌。
因此，我的下一步行动将是跟随对手的All-in下注40。

Romeo 还是在继续自己的套路，让我感觉有点啰嗦。而且这次回复的内容几乎在复读上一次的回复。

鉴于双方在 pre-flop 就已经 all-in，胜负其实也没有什么意义了。

总结

这次尝试借鉴了 generative agent 的想法，虽然简陋很多，但足够有趣，尝试的过程中带给了我很多欢乐。

从这次的尝试可以看出来在合适的 prompt 驱动下，gpt-3.5-turbo 也还不错，体现出了一定的理性。

这次尝试还让我有几个后续的想法：

整理一个 github repo 把这些 chat 公开出来
用 langchain 来自动化这个过程，做一个游戏出来，到时候就不是围观 AI 打德扑，而是和 AI 同台竞技啦
用其他模型（如Claude-Instant，vicuna-13b）替换 ChatGPT 跑跑看，这个其实已经在做了:）
