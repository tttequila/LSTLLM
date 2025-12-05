## 文章收集
### Long-Term
1. Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
2. Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning 
3. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
4. Mem-α: Learning Memory Construction via Reinforcement Learning

### Short-Term
1. MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent
2. CAMELoT: Towards Large Language Models with Training-Free Consolidated Associative Memory

### Dataset
1. MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems
2. Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions



# Benchmark设置

#### MemoryBench（Ai et al., 2025）

[Hugging face](https://huggingface.co/datasets/THUIR/MemoryBench)

→ **系统级记忆测试基准，任务包含长上下文、人物设定、事实更新与回忆**

- **用途：** 测试长期记忆能力与更新一致性
- **拆分方向：**
  - 记忆插入/更新场景 → 训练长期记忆 Agent（CRUD）
  - 人设类 QA → 训练事实拆分 + 长期记忆协同

#### Incremental Multi-Turn Benchmark（MemoryAgentBench）（Hu et al., 2025）

[Hugging face](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/viewer/default/Accurate_Retrieval?views%5B%5D=accurate_retrieval&views%5B%5D=test_time_learning&views%5B%5D=long_range_understanding&views%5B%5D=conflict_resolution&sql=%28SELECT+%27accurate_retrieval%27+AS+subset%2C+*+FROM+accurate_retrieval+LIMIT+2%29%0AUNION+BY+NAME%0A%28SELECT+%27long_range_understanding%27+AS+subset%2C+*+FROM+long_range_understanding+LIMIT+2%29%0AUNION+BY+NAME%0A%28SELECT+%27test_time_learning%27+AS+subset%2C+*+FROM+test_time_learning+LIMIT+2%29&sql_row=0)

→ **增量式交互记忆基准**，专门测试 memory update 与使用能力（与 Mem-RL 架构强相关）

- **用途：** 模拟真实对话中的「信息积累 - 更新 - 访问」
- **拆分方向：**
  - 用于长期记忆 Agent 的增量学习阶段（update / delete 学习）
  - 用于 RL 微调（reward = consistency + correctness）



## 长短期记忆框架

1. 事实拆分 Agent
	- 作用：将当前新的 Text 拆分成多个事实并判断每个事实是否重要
	- 输入：新 Text + 当前 Query
	- 输出：m 条重要事实和 n 条不重要事实
2. 长期记忆 Agent
	- 作用：构建 RAG
	- 输入：m 条重要事实，每条事实检索到的 Top-k 记忆
	- 输出：对每一条事实从四个选项[add、update、delete、noop]中选一个，并输出对应的操作结果，用来更新 RAG（参考 Memory-R1 图2 和 P11）
	- 附加：如果一条重要事实在这里被选为 noop 操作，那应该给事实拆分 Agent 一个负反馈。
- 短期记忆 Agent：
	- 作用：构建Memory state，将不重要的事实进行压缩存储。
	- 输入：历史 Memory state + 不重要事实
	- 输出：新的 Memory state 
- 生成 Agent
	- 作用：根据 Memory state、检索到的 RAG、Query 进行作答

# TODOs

总共涉及六个模块：

1. 模型与Verl后端配置层 --> 完全由verl提供，不需要管
2. Memory储存和检索 --> 可以基于 mem-$\alpha$ 的`memory.py` 进行修改
3. Memory工具接口 --> 在 mem-$\alpha$ 的 `functions.py` 的基础上，接上verl的 `BaseTool` 类，复用`BaseTool`的相关的接口
4. Rollout --> 借鉴 mem-$\alpha$ `generation.py` 的思路，自己实现 rollout 流程
5. RL glue，reward pipeline --> 依旧借鉴 mem-$\alpha$ 的整体数据流
6. Evaluation

## 后端配置

- `actor_rollout_ref.model.path` 指向你的 base LLM；
- `strategy=fsdp` / `fsdp2`；
- `actor_rollout_ref.model.lora_rank` 等 LoRA 参数；
- `actor_rollout_ref.rollout` 选 vLLM async 模式

## Memory 环境

- 提供一个统一的 `Memory` 类/模块，负责：
  - 长期记忆：semantic / episodic store（向量索引 + metadata）；
  - 短期记忆：compressed summary / 临时 notes；
  - 提供基础 API：
    - `insert(memory_type, content, meta)`
    - `update(id, new_content)`
    - `delete(id)`
    - `search(query, top_k)` 等
- 实现检索逻辑（BM25 + embedding）；
- 制定不同模式下的 **system prompt 渲染**：
  - 记忆操作模式（memorie mode）
  - 普通对话 / QA 模式（chat mode）
  - 记忆压缩 / consolidation 模式（rethink mode）

**基本可以复用`memory.py` 几乎就是这层的现成实现**

## 工具接口

- 定义一组「工具」对应 memory 操作，比如：
  - `new_memory_insert`
  - `memory_update`
  - `memory_delete`
  - `search_memory`
- 每个工具需要提供：
  - 参数 schema（给模型看的 OpenAI/JSON schema）；
  - 真正的执行逻辑（调用第 2 层 `Memory` 的 API）；
  - （可选）工具级奖励/指标（比如记录写入条数、压缩率等）。

**可以复用**

- `functions.py` 已经有：
  - `Parameter` / `ToolFunction` 抽象；
  - `NewMemoryInsert` / `MemoryUpdate` / `MemoryDelete` / `SearchMemory` 四个类；
  - `FUNCTION_IMPLS`、`MEMORY_TOOL_SCHEMAS` 等映射。

## Rollout层

实现一个「一条训练样本 → 多步记忆 + 多轮对话 → 最终回答」的 rollout 过程

在代码层面，这一层需要实现：

- 一个类似 `MemoryGenerationManager` 的类，包含：
  - `run_memory_loop(batch)`：给一批样本，跑完整个 episode，返回 trajectories；
  - 内部的：
    - prompt 构建（调用 `Memory.render_system_prompt` + tools schema）；
    - 调用 vLLM rollout（`actor_rollout_wg.generate_sequences`）；
    - 解析 JSON function call（替代原来的 QwenFnCallPrompt + 特殊 token）；
    - 执行工具（调用第 3 层 API，更新 Memory）；
    - 记录所有 token / logprob / mask / 工具调用信息；
- 一些 helper：
  - JSON 修复与解析（可从 `agent.py` 里保留一小部分现在的 `json_repair` 使用方式）；
  - `TensorHelper` 封装 padding / 合并序列（复用 `generation.py` 里的实现）；
  - 用于多 GPU 的 padding/分发（`_generate_with_gpu_padding` 之类）。

**可以从 `generation.py` / `agent.py` 里复用的：**

- **强烈建议复用 / 适配：**
  - `TensorHelper` + 与 Verl `DataProto` 对接的那套 batch 处理逻辑；
  - chunk 循环、rolling state 的整体结构（即每个 chunk 的处理顺序）；
  - 对 response 的基本截断和 re-tokenize 模式（只是把 `<memory_insert>`/Qwen 特殊 token 改成 JSON）。
- **不建议原样照搬，需要重写的：**
  - 所有依赖 QwenFnCallPrompt / `✿FUNCTION✿` / `<think>...</think>` 的东西；
  - `process_text_with_qwen_pipeline` 那条链；
  - function call 的正则解析。
