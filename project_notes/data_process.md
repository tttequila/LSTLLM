# 对齐RLHFDataset

为适配多 agent / 分支 rollout，数据侧不再套 chat 模板，而是按 **question 粒度** 展开；每行携带原始文本与 per-agent prompt/plan，模板化与 padding 在自定义 rollout 内完成。

## 最低列要求（Parquet）
1. `prompt`（或通过 `prompt_key` 指向）：可为精简的 answer-agent system prompt 占位，不需预先拼 chunks/question。
2. `extra_info`（Dict），建议字段：
   - `chunks: List[str]`：原始 chunk 列表（不再拼成长字符串）。
   - `question: str`：当前 QA。
   - `answer_gt` / `answers`：参考答案。
   - `sample_id` / `question_idx` / `group_id`：追踪来源；训练分组仍用 fit 内生成的 `uid`。
   - `agent_prompts: Dict[role, {system, user, params}]`：各 agent 的原始 prompt 片段（未套模板）。
   - `agent_plan: List[str]`：执行顺序/依赖（如 `["fact_split","long_mem","short_mem","answer"]`）。
   - 可选 `chunk_window_meta` 等补充标签。

原则
- 数据层只存原始文本与 per-agent prompt/plan，不做 chat 模板、不做 tokenizer padding。
- collate_fn 只负责打包/（可选）展开 agent 维，不渲染模板。
- rollout 内根据 agent_plan/依赖动态组装 prompt，调用 tokenizer/vLLM，生成符合 Verl 规范的 `gen_batch_output`，再与原始 batch `union`。

---

`RLHFDataset`的主要逻辑以及参数包括:
> 下载/读取 → 过滤 → 应用 chat template → tokenizer/processor 生成 input_ids/attention_mask/position_ids 等张量，同时保留原始元信息（extra_info 等），方便上层直接构建  
> 这里我们仅复用其加载/缓存能力，chat template 和 padding 由自定义 rollout 负责。

- **data_files (str or list)** – Path(s) to Parquet file(s).
- **tokenizer (PreTrainedTokenizer)** – For the tokenization of text to token IDs.
- **config (DictConfig)** – cache_dir, prompt_key, max_prompt_length, truncation, etc.
- **processor (ProcessorMixin, optional)** – multimodal preprocessor.

---

## MemoryAgentBench → QA级样本

目标：在进入 Verl 前就把 MemoryAgentBench 洗成 “QA 粒度” 的行，从而让 GRPO 的组内扩增直接作用在 question 级 episode 上。

1. **拆分粒度**
   - 每条原始实例可能包含 `chunks` + 多个 `questions_and_answers`。
   - 预处理时将其展开：每条 QA 生成一行，带上同一份 `chunks`（或预裁剪 window）以及唯一 `group_id = f"{sample_id}-q{idx}"`。
   - 这样 RayPPOTrainer 在最外层 repeat n 次时，会得到 `B × q × n` 条 episode。
   - ⚠️ 组号策略：数据清洗阶段可保留自定义 `sample_id/group_id` 作为追踪，但训练时沿用 Verl 在 fit 中生成的 `uid` 作为实际 GRPO 分组，自定义 id 仅作日志。

2. **列设计（更新版）**
   - `prompt`：精简的 answer-agent system prompt 占位；模板由 rollout 渲染。
   - `extra_info` 推荐字段：
     - `chunks`: List[str]
     - `question`: 当前 QA
     - `answers`: 参考答案
     - `sample_id` / `question_idx` / `group_id`
     - `chunk_window_meta`: 可选
     - `agent_prompts`: 各 agent 原始 prompt 片段（未模板化）
     - `agent_plan`: agent 执行顺序/依赖

3. **兼容 RLHFDataset**
   - 以上字段均可写入 Parquet；`prompt_key` 指向 `prompt`，`extra_info_key` 指向 `extra_info`。
   - collate_fn 可在需要时将 agent 维与 batch 维展开为 `[batch*agents, ...]`，但不渲染模板、不做 tokenizer padding；渲染与 padding 在 rollout 侧完成。

4. **rollout 配合**
   - CustomRolloutWorker 拿到 QA 行后，先遍历 `chunks` 执行事实拆分 → 长期记忆写入 → 短期压缩，再切换到最终 question 生成。
   - reward 以 QA 粒度计算，并广播给该 episode 的生成 token。
   - 如需同时训练“仅记忆管理”的子任务，可在同一 extra_info 中附加 flag，由 rollout 选择模式。
   - 多 agent 并列（事实拆分/长记/短记等）时，可在 collate/采样后将 agent 维展开到 batch 维，`agent_role` 写入 `non_tensor_batch`，rollout 渲染对应模板，输出符合 Verl 规范的轨迹并统一更新。

---

（最后更新：2025-12-11）