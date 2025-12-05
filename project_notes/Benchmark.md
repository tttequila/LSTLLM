# MemoryAgentBench (Hu, etc., 2025)

所有样本都由以下两个部分构成：

1. chunks

   $c_1, c_2, c_3, \cdots, c_n$ 一条chunk sequence，将长文本拆解成多个chunk，顺序输入给LLM，并配上prompt让模型记住chunk里面的内容

2. QA pairs

   $(q_1, a_1), (q_2, a_2), \cdots, (q_m, a_m)$ 一组问答对。每个chunk序列都会配给多个问答对，但是**问答对是在模型记忆完chunk序列之后，统一进行问答的**。

本质上是一个增量的chunk序列，而不是真正意义上的multi-turn conversation，更像是：

```yaml
Turn 1: 用户发送 chunk1 + “请记住这部分”
Turn 2: 用户发送 chunk2 + “请记住这部分”
...
Turn N: 用户发送 chunkN + “请记住这部分”
```

agent在整个过程中只能累积地记忆这些chunks。整个测评流程如下：

#### 阶段1：增量记忆chunks

```yaml
User: c1 （记住）
Agent: ok
User: c2 （继续记住）
Agent: ok
...
User: cn
Agent: ok
```

#### 阶段2：提问阶段

```yaml
User: q1
Agent: a1_pred
User: q2
Agent: a2_pred
...
```

重点在于memory的**增量吸收**，而不是在 generally **在对话过程当中更新记忆库**。

### 数据子集

数据由四个子集构成：

1. Accurate Retrieval
2. Test Time Learning
3. Long Range Understanding
4. Conflict Resolution （Selective Forgetting）

结构上如之前提及的两部分组成：

```json
{
    "chunks": [
        {"chunk_id": 1, "text": "... long context part 1 ..."},
        {"chunk_id": 2, "text": "... long context part 2 ..."},
        ...
    ],
    "questions": [
        {"qid": 1, "question": "...?"},
        ...
    ],
    "answers": [...]
}
```

# MemoryBench (Ai, etc., 2015)

MemoryBench的每条数据本质上是 一个任务样本 $(q, c, v)$， 即*用户指令+任务上下文+答案/打分准则*，和一段多轮交互日志 dialog + 一段隐式反馈日志 implicit feedback

Huggingface上的统一字段结构如图：

```json
{
  "test_idx": 456,		# 唯一标识符，对于一些长上下文相关的数据条目，这个字段同样也指向了corpus目录下，相关的文档
  "input_prompt": ...,	# 可能是“input_prompt”：直接就是raw prompt
						# 也可能是“input_chat_messgaes”：使用了chat模板的prompt（列表，每个元素都有role和content）
  "dataset_name": "NFCats",
  "lang": "en",
  "info": {},			# evaluation需要的信息
  "dialog": [			# 训练阶段，用Qwen3-8B作为LLMsys，Qwen-32B作为User Simulator得到的对话日志
    {
      "content": "...user message...",
      "role": "user"
    },
    {
      "content": "...assistant answer...",
      "role": "assistant"
    }
  ],
  "implicit_feedback": [	# 训练阶段产生的隐式日志，包括满意度打分，[like, copy, dislike]等行为，对话轮次，是否结束等
    {
      "implicit_actions": [],
      "round": 1,
      "satisfaction_score": 8,
      "terminated": true
    }
  ]
}
```

`input_prompt`可以理解成当前的query，而`dialog`则更像是用于训练的数据或者历史数据。整个benchmark的目标就是测试在已有历史经验（`dialog`+`implicit_feedback`）的基础上，模型能不能再`input_prompt`上表现得更好

## 子任务类别

1. 长对话任务：以LoCoMo/DialSim为例
2. 写作/生成任务：以WritingBench，HelloBench，JRE-L，IdeaBench为例
3. 短文本任务：以LexEval/NFCats/some JUGE task为例