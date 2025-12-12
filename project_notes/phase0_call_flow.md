# Phase 0 调用流（RayPPOTrainer → 自研多轮 Rollout）

> 目标：记录我们如何在 `RayPPOTrainer.fit()` 内接入 Phase 0 的多轮 rollout，并明确数据在各模块之间的流动。该文档会随着实现推进及时更新。

---

## 总览（更新）

```
RayPPOTrainer.fit
  ├─ dataloader → DialogueBatch（符合统一规范的样本）
  ├─ adapter/collate → List[BatchInput]（可在此展开 agent 维，但不渲染模板）
  ├─ CustomRolloutWorker.run_batch(episodes)
  │     ├─ EpisodeRunner.run_episode()
  │     │     ├─ history turns: 事实拆分/长期记忆写入 + 短期压缩（可多 agent）
  │     │     └─ target turn: 按 agent_plan 渲染 prompt→generate→memory update
  │     └─ TrajectoryBatchBuilder.build()  # step-level pad，奖励广播，保留 step meta/role
  ├─ PackedBatch → DataProto.from_dict(...)（包含 agent_role/uid/group_id 侧信息）
  └─ core_algo.update(data_proto)  # GRPO/PPO 更新 LoRA
```

---

## QA级 Episode 设计（多 agent 版）

- **数据粒度**：上游数据洗成“(chunk 序列, question, answer)”的 QA 子样本，已按 QA 展开。
- **rollout 输入**：BatchInput 携带 `agent_prompts`/`agent_plan`，rollout 按 plan 渲染每个 agent 的 prompt，再产出多条独立轨迹。
- **GRPO 行为**：训练分组仍用 fit 中生成的 `uid`；自定义 `sample_id/group_id` 仅作追踪。
- **多 agent 并列**：可在 adapter/collate 阶段展开 agent 维，再与 batch 维 flatten 成 `[batch*agents, ...]`；`agent_role` 作为 side meta，避免不同角色梯度混杂。

---

## 关键步骤

1. **RayPPOTrainer.fit**
   - 原本会调用 `actor_rollout_wg.generate_sequences`；Phase 0 中我们在同一个位置改用 `CustomRolloutWorker.run_batch`。
   - 输入：dataloader 产出的原始 batch（dict/Tensors）以及 tokenizer。

2. **Dataset Adapter → BatchInput**
   - 每条 **QA 子样本** 转成 `BatchInput`：
     - `episode_id`/`group_id`
     - `turns: List[TurnSpec]`，区分：
       - `turn_role=history`：现成对话历史，仅供记忆处理
       - `turn_role=target`：最后一轮提问，需模型生成+训练
     - `final_query`、`target_answer`、自定义 `metadata`
     - `agent_prompts` / `agent_plan` / `agent_role`
   - 不在此阶段渲染 chat 模板；模板化与 padding 留到 rollout。
   - 这一步保证所有 benchmark 都喂给 rollout 相同的结构。

3. **CustomRolloutWorker.run_batch(batch_inputs)**
   - 依次调用 `EpisodeRunner.run_batch_inputs`，返回 `EpisodeTrajectory` 列表。
   - `EpisodeRunner` 在历史轮跳过回答，只调用记忆相关 agent；在目标轮按 agent_plan 渲染 prompt，调用 `PolicyClient.generate` 与 logprob 计算。
   - 作用类似 Mem-alpha 的 `MemoryGenerationManager.run_memory_loop`，但拆成模块化组件。

4. **EpisodeRunner.run_episode**
   - 循环所有 turn：
     1. `turn_role=history`：
        - `MemoryManager.extract_facts`：事实拆分 + 长期记忆写入，记录写入的 fact id。
        - `MemoryManager.compress_short_term`：对近期对话做短期记忆压缩，记录摘要 id。
        - 可选：记录检索/写入日志到 `memory_ops`，预留 step-wise 奖励占位。
     2. `turn_role=target`：
        - `PromptBuilder.build_turn_prompt` 注入长期检索结果 + 短期摘要。
        - `PolicyClient.generate`（vLLM/FSDP 等）返回 `GenerationOutput`。此阶段同时负责 QA agent 的回答与记忆 agent 的回读。
        - `StepTrajectory.from_generation` 收集 token/logprob/mask。
        - `MemoryManager.update_memory` 用生成结果更新记忆。
   - Episode 结束后用 `RewardFn` 计算 episode 级 scalar reward（用于策略），写进 `EpisodeTrajectory`；逐步奖励留给记忆 agent 后处理。

5. **TrajectoryBatchBuilder.build**
   - 仅将需训练的 agent 轨迹（通常 target 轮的 answer 或指定角色）纳入训练张量；其他可置 `response_mask=0` 或丢弃。
   - step-level flatten & pad 到 `[num_steps, max_seq_len]`，episode scalar reward 广播到对应 steps。
   - 返回 `step_meta`/`step_rewards_ext`，并附带 `agent_role`/`turn_role`。
   - 输出 `PackedBatch`（`input_ids`, `logprobs`, `response_mask`, `rewards`, `group_ids`, `episode_ids`, ...）以及 sidecar。

6. **封装为 DataProto 并执行优化**
   - 将 `PackedBatch` + 需要的非 tensor 元数据（uid、原始文本、step_meta/step_rewards_ext）包装成 `DataProto`（sidecar 字段）。
   - 交给 `core_algo.update`（GRPO/PPO）完成 actor/critic/ref 的反向更新，与标准 Verl 训练流程一致。

---

## 接入提示

- **替换点**：`RayPPOTrainer.fit` 中 `generate_sequences` 段落改为 Phase0 rollout，返回的 `PackedBatch` 填充到 `batch.batch`。
- **Group/Reward 对齐**：训练分组用 Verl 生成的 `uid`；自定义 id 仅追踪。reward 为 episode 级 scalar，广播到该 episode 的 steps。
- **QA 粒度 + agent 展开**：QA 已展开，组内 repeat 直接得到 B×q×n；如有多 agent，在 pack 前展开 agent 维到 batch 维。
- **模板与 padding 位置**：不在 adapter/collate 做模板化；rollout 内按 agent_plan 渲染并 padding，保证 `gen_batch_output` 满足 Verl 规范。
- **逐步奖励与过滤**：历史轮或非训练角色可置 `response_mask=0`；`step_rewards_ext` 预留给记忆步骤的后处理。
- **扩展性**：Prompt/记忆写入/Policy 后端均接口化，后续 Phase1/2 替换实现无需动 RayPPOTrainer 主流程。

---

（最后更新：2025-12-11）

