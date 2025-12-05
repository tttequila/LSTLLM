# Memory模组

集中管理在RAM当中，存在三类记忆：

1. core memory：通常直接使用字符串
2. semantic：由一个字典列表进行管理
   1. self.semantic：记忆本体
   2. self.semantic_embedding_matrix：记忆速查矩阵，初始为[0，1536]的空矩阵
   3. semantic_embedding_ids：记忆ID
3. episodic：同semantic

## Helper函数

- `total_length`：统计记忆库总共包含夺少token的记忆
- `_generate_memory_id`：将uuid的前四位作为唯一标识符
- `_content_exists`：检测该记忆模块中是否已经存在`content`相关的记忆
- `_get_embedding`：调用openai的text-embedding-3-small模型生成当前内容的embedding

## Render

- `render_system_prompt`：根据工作模式不同，拼接prompt

  ​	先从semantic和episodic记忆的List[Dict]取出后`max_num_of_recent_chunks`数量的记忆，作为`*_items`，根据是否使用core memory将所有`*_items`拼成三元组的`memory blocks`，这个`memory blocks`就是当前时间步的`memory state`

  1. `status=memorie`：直接取出三个模块的中对应记忆的数量，放入对应的特殊token之间，拼入chat模板返回

  2. `status=rethink`：直接将整个`memory state`丢给LLM进行重新整理，要求模型按照以下要求：

     - 冗余清理：检查并移除重复、占位或表述不清的记忆，同时保持关键信息。


     - 信息综合：基于已有记忆推导新的事实或总结，把洞见写回记忆。


     - 结构优化：梳理不同记忆之间的关系，让记忆更有条理。


  积极调用 `memory_delete`、`memory_update`、`memory_insert` 等函数，删除旧的、更新现有的、添加新总结。

  3. `status=chat`：先进行判断，当前拿到的`semantic_items`是不是全量的。如果是全量的则告诉LLM这就是全部记忆，如果不是，则提醒LLM可以调用`search_memory`操作，基于当前query再次查询整体记忆库。最终期望模型返回response，可能是重新查询指令，也可能是最终的回答


  ​    

- `_block`：将记忆包装上前后缀特殊token，包裹成块，在render_system_prompt中被调用

## Memory Operations

所有的memory操作都会被functions.py调用，这部分主要是定义了几种操作

- `new_memory_insert`：插入新的记忆，接受*记忆类型*和*内容*两个参数：
  - 如果是`core`，则根据不同的情况，返回报错告诉模型`core memory`没被启用/没有初始化/不可以插入新记忆请使用`memory_update`
  
  - 如果是其他部分记忆，则优先查重，记忆重复则返回None跳过，否则生成新记忆的id和embedding，更新matrix和ids并返回`{memory_id:content}`
  
- `memory_update`：修改已有的记忆，接受*记忆类型*，*新内容*以及*记忆id*三个参数
  - 如果是`core`，则整块替换，超过512 tokens自动截断并附加截断提示
  - 其他记忆类型则直接遍历替换，更新matrix和ids，返回更新之后的`{memory_id:content}`
- `memory_delete`：删除记忆
  - 与insert类似，根据不同情况返回不同报错，但是允许直接delete core memory
  - 其他记忆类型则移除对应条目，matrix以及id list，如果是不存在的ID则打印警告但不终端
- `memory_search`：检索
  - `core`报错
  - `semantic`和`episodic`则使用beam25或者text embedding检索
    - `bm25`直接使用`rank_bm25`算法库
    - `text embedding`通过批处理计算

# 训练流程实现

`main_ppo.py`内部没有明显改动，主要是在`TaskRunner`里加入了记忆库相关的组件以及循环，并且直接在`.run()`方法下显式使用了记忆库组件，而没有加入到`AgentLoop`类当中（但应该不影响）

主要的改动发生`RayPPOTrainer.fit() line 1152`开始：

0. 基础训练配置：

    - algorithm.adv_estimator=grpo
    - actor_rollout_ref.actor.use_kl_loss=true
    - kl_loss_coef=1e-3，并关掉 in-reward KL
    - customized_grpo_rollout_n 提到 8，让每个 prompt 一次 rollout 8 条轨迹用于组内归一、把批量 train_batch_size 降到 32，prompt/response 长度放大到 4096/2048
    - use_memory_mode=true, do_search=true，指向外部工具服务

    RayPPOTrainer内部会把`use_crtic`设置为`False`，训练的时候不会创建 `critic worker`

1. 初始化一个`MemoryGenerationConfig`作为记忆库相关的配置文件，以及`MemoryGenerationManager`的管理组件

   > `MemoryGenerationManager`主要接受三个输入：tokenizer，actor_rollout_wg以及config。所以尽管主训练循环基本发生在`run_memory_loop`内，所使用的模型仍然是verl框架下的actors

2. 在每个 step 中先从 dataloader 取出 batch，构造 `gen_batch`、`chunks`、`questions_and_answers`、`data_sources` 后`trainer.customized_grpo_rollout_n=8` 会把每个 prompt、chunk、QA 成倍复制，确保同一 prompt 下有 8 条 rollout 供 GRPO 组内对比

   > `ray_trainer.py` 1212-1258

3. 调用 `MemoryGenerationManager.run_memory_loop` 生成这一批 rollouts；该函数会为每个样本创建对应的 Memory，并在循环里借助 `MemoryAgent` 的逻辑执行记忆函数调用、收集工具调用奖励等元信息。

   > `ray_trainer.py`1252-1257
   >
   > `generation.py` 349-476
   >
   > **`generation.MemoryGenerationManager.run_memory_loop()`**
   >
   > 0. 读取batch数据，创建并维护一个`active_mask`用来追踪这个batch内哪些chunk被使用了哪些没有
   >
   > 1. 为每个数据实例单独创建一个`Memory`类，维护`chunk_inputs_ids`/`*_response_ids`/`*_response_mask`/`*_function_call_rewards`/`*_function_calls`等追踪训练相关数据
   >
   >    （开始遍历循环👇）
   >
   > 2. 遍历每个数据实例当中当前位置的chunk，将数据实例批中的当前index的chunk置入`current_chunks`作为当前批次当前时间步仍需处理chunks，初始化一个滚动状态，调用`self._process_chunk_with_memory_operations`函数
   >
   >    > `self._process_chunk_with_memory_operations`函数接受三个输入：
   >    >
   >    > 1. rolling：当前批次再进入本轮chunk前的滚动状态（输入token，attention mask，position id等）
   >    > 2. current_chunks：List[str]，表示当前批次中仍在处理的已套用prompt模板的chunk样本
   >    > 3. batch_memory：List[Memory]，与current_chunks对齐的Memory对象，用于再函数调用的时候提供具体的记忆库操作环境
   >    >
   >    > 该函数会创建一个MemoryAgent类作为渲染模板使用，然后将该批次的chunk送给MemoryAgent套用模板之后交给`self._process_next_chunk`进一步调用qwen官方的fncall处理函数进行agent response generation，并执行function call。清理记忆functions以及结果相关的部分（因为记忆操作和生成用的是同一套模板，而现阶段我们只想保留记忆操作相关的部分进行训练）
   >    >
   >    > 返回四个值：
   >    >
   >    > 1. chunk_inputs_ids：每个实例再本轮此输入给模型的prompt token id序列，就是本轮次调用里LLM收到的prompt batch
   >    > 2. response_ids：batch形式的response
   >    > 3. response_mask：label mask
   >    > 4. updated_meta_info：Dict，包含函数调用平均成功率，调用详情的信息，后续可以用来计算奖励并记录日志
   >
   > 3. 同时，将所有记忆和问题发送至外部大模型获取回答，收集基于MemoryAgent提取的记忆进行RAG的答案生成（整个项目训练的是一个能“决定记什么，何时记，如何更新”的记忆管理策略，答案完全由冻结的外部模型+提取的记忆进行RAG）。外部冻结模型的主要作用在于计算RL reward
   >
   > 4. 最后对于batch内的每个数据实例，拼装出最终的输出（prompt+response得到input_ids，以及attention mask和position ids，并将response写回末尾确保只有中间的memory agent部分参与训练）。补上function call content reward。
   >
   > 5. 打包meta_info
   >
   > 6. 返回final output
   >
   > 最终`run_memory_loop`返回的`DataProto`内容为：
   >
   > >  **主张量字段**
   > >
   > > - **prompts**：堆叠后的 chunk 输入 token，所有 chunk 展平后左侧 pad；形状 [num_chunks_total, prompt_len]。
   > > - **responses**：对应的模型输出 token（截断清理过），右侧 pad；形状 [num_chunks_total, response_len]。
   > > - **response_mask**：布尔掩码，标记 responses 中哪些位置需要计算损失（补的结束语位置会是 0）。
   > > - **input_ids**：prompts 与 responses 拼接后的整体序列 [num_chunks_total, prompt_len+response_len]。
   > > - **attention_mask**：同形状的 0/1 掩码，最后一段被替换为 response_mask，确保仅在生成段计算梯度。
   > > - **position_ids**：由 TensorHelper 生成的位置编码。
   > >
   > > **meta_info 字段**
   > >
   > > - **questions_list、predicted_answers_list、ground_truth_answers_list**：外部问答服务返回的问句、预测答案与对应真值。
   > > - **indices_in_batch**：每个条目对应的原始 batch 索引（处理多个 chunk 时保持映射）。
   > > - **total_chunk_length、total_memory_length**：每个样本累计的 chunk token 数、最终记忆条目长度。
   > > - **every_chunk_length**：每条生成响应的有效长度统计。
   > > - **batch_memories**：最终记忆内容的快照（core/episodic/semantic）。
   > > - **all_function_call_rewards**：每条样本的函数调用成功率。
   > > - **all_function_calls**：每次调用的详细记录（名称、参数、执行结果、是否成功）。
   > > - 若启用了 analyze_function_url，还会有 all_function_call_content_rewards。
   > > - 以及最近一个 chunk 的其他调试信息（由于 final_output.meta_info.update(last_chunk_meta_info) 保留了生成器输出的元数据）。

4. `run_memory_loop` 的输出被拼装回 `DataProto`，其中 `prompts`/ `responses`/ `response_mask` 供策略更新用，而 `meta_info` 中记录的 `function_call_rewards`、`indices_in_batch` 等则用于后续的奖励统计与监控。函数末尾还会按 GPU 数量补齐样本，并把记忆长度、问题回答等附加到 `meta_info`

   > `generation.py` 449-593

5. 生成后的 `final_gen_batch_output` 与原始 batch 合并：先补齐 `response_mask`、均衡 DP rank，再写入 `global_token_num` 等统计；随后进入奖励阶段。若配置了 RM，则调用 `rm_wg.compute_rm_score`，之后用同步或异步方式运行统一的 `reward_fn`，把准确率、压缩率、函数调用成功率等指标记录在 `reward_extra_infos_dict`

   > `ray_trainer.py` 1328-1369

6. 因为是 GRPO，优势直接基于组内得分归一化，不需要 value network；返回的 advantages 与 returns 同为归一后的分数，计算KL约束等进行更新（该部分为verl的内置模组）

   > `core_algos.py` 199-254
   >
   > `ray_trainer.py` 1384-1478

# Agent模组

只定义了一个`MemoryAgent`类，主要作为交互环境存在，被`MemoryGenerationManager`当作模板工具箱，复用消息预处理于函数调用解析，简单来说`run_memory_loop`函数在进入推理钱，先用`MemoryAgent.process_text_with_qwen_pipeline`把每个数据块渲染上记忆，同时一个管理器在rollout结束之后通过`memory_agent_template`调用`MemoryAgent._parse_response`与`_run_tool_from_function_call`，从actor的response里抽取functions并执行，打分，把奖励写回meta info

简单来说，就是环境的一部分。

# Functions模组

## `ToolFunction`

所有function的基类，并提供统一的`execute(memory, argument)`和`to_schema(memory)`给外部进行调用，约定必须声明`name`，`description`以及`parameters`成员变量