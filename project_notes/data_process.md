# 对齐RLHFDataset

`RLHFDataset`是verl的标准化数据集类,数据要对齐`RLHFDataset`类,需要将数据洗成至少包含以下两个列的parquet文件

1. `prompt` (也可以在配置文件里通过`prompt_key`指向某一列): 必须是列表形式的对话轮次,每条消息是 `{"role": "...", "content": "..."}`
    - 角色至少包含 user 与 assistant，可以加 system、tool 等。
    - 历史轮与目标轮仍然顺序排列在同一个 list 里，后续可以在 extra_info 内标记 turn_role。
    > Example:
    > ```json
    > {
    >  "prompt": [
    >    {"role": "system", "content": "你是..."},
    >    {"role": "user", "content": "历史轮1观察..."},
    >    {"role": "assistant", "content": "历史轮1回应..."},
    >    {"role": "user", "content": "历史轮2观察..."},
    >    {"role": "assistant", "content": "历史轮2回应..."},
    >    {"role": "user", "content": "最终问题..."}
    >  ],
    >  "extra_info": {
    >    "group_id": "g123",
    >    "episode_id": "e456",
    >    "turn_roles": ["history","history","history","history","history","target"],
    >    "turn_metadata": [{}, {}, {}, {}, {}, {"note": "final"}],
    >    "final_query": "最终问题...",
    >    "target_answer": "参考答案（可选）"
    >  }
    > }
    > ```
2. `extra_info`: 字典类型,通常在后续流程中进行额外操作

---

`RLHFDataset`的主要逻辑以及参数包括:
> 下载/读取 → 过滤 → 应用 chat template → tokenizer/processor 生成 input_ids/attention_mask/position_ids 等张量，同时保留原始元信息（extra_info 等），方便上层直接构建
- **data_files (str or list)** – Path(s) to Parquet file(s).

- **tokenizer (PreTrainedTokenizer)** – For the tokenization of text to token IDs.

- **config (DictConfig)** – Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.

- **processor (ProcessorMixin, optional)** – Multimodal preprocessor for images/videos.

---

## MemoryAgentBench → QA级样本

> 目标：在进入 Verl 前就把 MemoryAgentBench 洗成 “QA 粒度” 的行，从而让 GRPO 的组内扩增直接作用在 question 级 episode 上。

1. **拆分粒度**
   - 每条原始实例可能包含 `chunks` + 多个 `questions_and_answers`。
   - 预处理时将其展开：每条 QA 生成一行，带上同一份 `chunks`（或预裁剪 window）以及唯一 `group_id = f"{sample_id}-q{idx}"`。
   - 这样 RayPPOTrainer 在最外层 repeat n 次时，会得到 `B × q × n` 条 episode。

2. **列设计**
   - `prompt`：仍采用统一 system 指令 + 最终 user 提问。历史 chunk 不直接写进 prompt，而是放入 `extra_info`，由 rollout 自行 replay。
   - `extra_info` 约定字段：
     - `chunks`: List[str]，供事实拆分/记忆 agent 重放。
     - `question`: 当前 QA 的完整提问。
     - `answers`: 参考答案（list or str），reward 计算所需。
     - `sample_id` / `question_idx` / `group_id`: 追踪原始来源与 GRPO 分组。
     - `chunk_window_meta`: 可选，记录 chunk 时间窗或 source 标签，方便 rollout 做窗口化。

3. **rollout 配合**
   - CustomRolloutWorker 拿到 QA 行后，先遍历 `chunks` 执行事实拆分 → 长期记忆写入 → 短期压缩，再切换到 `prompt` 的最终 user 问题。
   - reward 以 QA 粒度计算，并广播给该 episode 的生成 token。
   - 如需同时训练“仅记忆管理”的子任务，可在同一 extra_info 中附加 flag，由 rollout 选择不同模式。

4. **兼容 RLHFDataset**
   - 以上字段均可写入 Parquet；`prompt_key` 指向 `prompt`，`extra_info_key` 指向 `extra_info`。
   - `group_id` 放在 `extra_info` 内，供 `RLHFDataset` 在 collate 阶段拷贝到张量批次中。

---

（最后更新：2025-12-06）