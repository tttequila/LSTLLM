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

1. **Accurate Retrieval**

   这个任务子集主要包含1. eventqa\_\*；2. longmemeval_s\*；3.ruler_qa1/qa2\_\*几种数据来源

   1. **eventqa\_\***：

      - context字段是一个超长字符串文本，通常是长文档的原始文本
      - questions字段是提问的prompt文本列表
      - answers字段则包含答案
      - metadata里的`previous_events`字段提供了一些前序事件，不过通常已经包含在question prompt里了

   2. **longmemeval_***：

      - context字段时间戳和对话交替的列表

        - 时间戳的格式类似于Chat Time: 2022/11/17 (Thu) 12:04
        - 对话部分则是套用了user-assistant对话模板的交替对话，长度不固定

      - question字段是自带时间戳文本的提问

      - answers字段则是正确答案

      - metadata的`haystack_session`字段提供了RAG查询结果，会返回一系列对话sessions，并且每个对话会被标记上是否包含正确答案，格式是一个三层嵌套的list，第一层元素对应每一个问题，第二层元素对应这个问题查询到的对话session，第三层则是这个session里面的对话turns，以下是session的例子：

        ```json
        [{'content': "...",
           'has_answer': False,
           'role': 'user'},
          {'content': '...',
           'has_answer': False,
           'role': 'assistant'},
          {'content': "...",
           'has_answer': True,			# 标记该对话轮次里出现了问题相关的答案
           'role': 'user'},
          {'content': '...',
           'has_answer': False,
           'role': 'assistant'},
         	...
          ],
        	...                                                                                            ]
        ```

        `question_types`字段里面有更细分的问题分类

   3. **ruler_***：

      - context字段提供了文档，格式类似`Document n:\n...\n\n`，即文档之间会用空行进行分隔
      - questions字段就是提问的文本
      - answers字段是答案关键词列表，一个问题可能有一到两个答案关键词

2. **Test Time Learning**

   主要数据来源是icl，recsys的数据不明确，建议直接过滤掉

   1. **icl_\***：
      - context字段是一个长文本，文本里面有复数个句子以及对应的数字label，label表示某种潜在分类
      - questions字段会给出句子列表
      - answer字段需要根据问题句子返回最接近的句子的label

3. **Long Range Understanding**

   主要数据来源是1. infbench\_\*；2. detective\_\*

   1. **infbench\_\***：主要是长文本总结任务
      - context字段是一个超长文本，段落之间会用两个空行进行分隔
      - questions字段是**一个**带shot的总结指令
      - answers字段则是标准答案
      - metadata里的`keypoints`字段会提供一些总结关键词
   2. **detective\_\***：主要是长文本的MCQA任务而不是总结
      - context字段是带段落号的超长文本
      - questions字段是多个带shot的MCQA
      - answer字段是具体的正确答案

4. **Conflict Resolution （Selective Forgetting）**

   主要数据来源是合成数据集 1. factconsolidation_mh/sh_* (sh: single-hop; mh: multi-hop)

   1. **factconsolidation_mh/sh_***:
      - context字段提供了多个事实条目
      - questions字段是问题列表
      - answer则是答案列表，目前来看虽然是嵌套list，但是每个问题只对应一个答案

统一的数据结构为：

```json
{
    "context": ...,
    "questions": [[q1], [q2], ..., [qn]],
    "answers": [[a1], [a2], ..., [an]],
	"metadata":{
        "demo", "haystack_sessions", "keypoints", "previous_events", "qa_pair_ids", "question_dates", "question_ids", "question_types", "source"
    }
}
```

---


## Mem-alpha的MemoryAgentBench处理流程

`process_memory_agent_bench()` 里确实按数据来源做了多层分类处理，来适配 MemoryAgentBench 各子任务的“杂乱”格式。

- **统一入口 + 初步过滤**：加载指定 split 后，先用 `metadata.source` 过滤掉 `niah`、`longmemeval_s_-1_500`、`eventqa_*`、`infbench_qa_eng_shots2`、`recsys_redial_full` 等超长或不适配的来源，确保后续逻辑只面对可控子集  

```3260:3288:Mem-alpha/process_data.py
if 'niah' in item['metadata'].get('source', ''): continue
if item['metadata']['source'] in [..., 'infbench_qa_eng_shots2']: continue
```

- **`longmemeval_s*` 专用通道**：对于这类“交替 timestamp + session 列表”的对话式 context，代码把每个 session 渲染为 `[Dialogue at timestamp ...]` 块：  
  - 逐 turn 校验结构，若单 turn 超长就用 `split_large_turn()` 先按句子/词切割，再由 `split_session_into_segments()` 组合成 ≤2048 token 的 segment；  
  - 组合 segment 时保持 `<User>/<Assistant>` 格式，并在 chunk 间拼接保证接近 2k token；  
  - 最终断言每个 chunk token 数合法  

```3300:3513:Mem-alpha/process_data.py
if item['metadata']['source'] == 'longmemeval_s*':
    ... render_session(...)
    ... split_large_turn(...)
    ... split_session_into_segments(...)
    assert count_tokens(chunk) <= 2048
```

- **TTL（Test_Time_Learning）**：先粗略按句子切块（max 1024），再把“Sentence … label …”重排成固定格式，套用 classification 模板 `[Dialogue between User and Assistant ...] + CLASSIFICATION_USER_TEMPLATES`，保证 few-shot 分类数据都有统一的句子+标签样式  

```3516:3671:Mem-alpha/process_data.py
elif split == 'Test_Time_Learning':
    chunks = create_chunks_use_sent_tokenizer(...1024)
    ...
    sentence,label = x.split("label:")
    chunk_content = f"Sentence: ...\nLabel: ..."
    formatted_chunk = f"[Dialogue ...]\n<User>{CLASSIFICATION_USER_TEMPLATES}\n{chunk_content}\n<Assistant>..."
```

- **其他 Accurate Retrieval 来源**：默认用 2048 token 句子切块，再根据 `source` 细分模板：  
  - `infbench_qa_eng_shots2` 走 BookSum 风格（事件日期 + `<System>` 提示）；  
  - 其余来源走“日期+User/Assistant 模板”；  
  - 所有 QA 原样保留（`question`,`answer`）  

```3577:3633:Mem-alpha/process_data.py
elif split == 'Accurate_Retrieval' and ... == 'infbench_qa_eng_shots2':
    formatted_chunk = f"[Event happened on ...]"
elif split == 'Accurate_Retrieval' and ...:
    formatted_chunk = f"[Dialogue between User and Assistant ...]"
```

- **Long Range Understanding**：无论 context 原始形式怎样，统一转成 BookSum 式 `[Event happened on ...]`，QA 改成“Summarize ... from start_date to end_date”，答案替换成预先抽取的关键词，附 `reading_dates` 供下游使用  

```3536:3575:Mem-alpha/process_data.py
if split == 'Long_Range_Understanding':
    formatted_chunk = f"[Event happened on ...]"
    updated_qa['question'] = "Summarize ... from {first_date} to {last_date}."
    updated_qa['answer'] = answers_to_keywords_mapping[str(item_idx)]['keywords']
```

- **统一输出**：无论哪类来源，最后都生成 `prompt + chunks + questions_and_answers` 的标准实例，并保留 `data_source`（accurate_retrieval/long_range_understanding）和 `sub_source`（原始 metadata.source），方便下游分别处理  

```3674:3689:Mem-alpha/process_data.py
data_instance = {
    'prompt': 'I will provide you with sequential information chunks...',
    'chunks': chunks,
    'questions_and_answers': questions_and_answers,
    'data_source': ...,
    'sub_source': item['metadata']['source']
}
```

所以，MemoryAgentBench 的“杂乱” context 会先按 `split` 与 `metadata.source` 分流：长对话（longmemeval_s*）→ turn 级切分；TTL → 句子+label 模板；书本/总结类 → BookSum 模板；普通检索 → User/Assistant 模板。这样可以把不同格式的原始数据统一到记忆代理可消费的 prompt 结构中。

---

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