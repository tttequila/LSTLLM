# Phase 0 调用流（RayPPOTrainer → 自研多轮 Rollout）

> 目标：记录我们如何在 `RayPPOTrainer.fit()` 内接入 Phase 0 的多轮 rollout，并明确数据在各模块之间的流动。该文档会随着实现推进及时更新。

---

## 总览

```
RayPPOTrainer.fit
  ├─ dataloader → DialogueBatch（符合统一规范的样本）
  ├─ adapter → List[BatchInput]
  ├─ CustomRolloutWorker.run_batch(episodes)
  │     ├─ EpisodeRunner.run_episode()
  │     │     ├─ history turns: 仅做事实拆分/长期记忆写入 + 短期压缩
  │     │     └─ target turn: prompt→generate→memory update
  │     └─ TrajectoryBatchBuilder.build()  # pad，奖励广播，保留 step meta
  ├─ PackedBatch → DataProto.from_dict(...)
  └─ core_algo.update(data_proto)  # GRPO/PPO 更新 LoRA
```

---

## 关键步骤

1. **RayPPOTrainer.fit**
   - 原本会调用 `actor_rollout_wg.generate_sequences`；Phase 0 中我们在同一个位置改用 `CustomRolloutWorker.run_batch`。
   - 输入：dataloader 产出的原始 batch（dict/Tensors）以及 tokenizer。

2. **Dataset Adapter → BatchInput**
   - 训练 batch 里的每条样本通过适配器转成 `BatchInput`：
     - `episode_id`/`group_id`
     - `turns: List[TurnSpec]`，区分：
       - `turn_role=history`：现成对话历史，仅供记忆处理
       - `turn_role=target`：最后一轮提问，需模型生成+训练
     - `final_query`、`target_answer`、自定义 `metadata`
   - 这一步保证所有 benchmark 都喂给 rollout 相同的结构。

3. **CustomRolloutWorker.run_batch(batch_inputs)**
   - 依次调用 `EpisodeRunner.run_batch_inputs`，返回 `EpisodeTrajectory` 列表。
   - `EpisodeRunner` 在历史轮跳过模型生成，仅调用记忆相关 agent；在目标轮才调用 `PolicyClient.generate` 与 logprob 计算。
   - 作用类似 Mem-alpha 的 `MemoryGenerationManager.run_memory_loop`，但拆成模块化组件。

4. **EpisodeRunner.run_episode**
   - 循环所有 turn：
     1. `turn_role=history`：
        - `MemoryManager.extract_facts`：事实拆分 + 长期记忆写入，记录写入的 fact id。
        - `MemoryManager.compress_short_term`：对近期对话做短期记忆压缩，记录摘要 id。
        - 可选：记录检索/写入日志到 `memory_ops`，预留 step-wise 奖励占位。
     2. `turn_role=target`：
        - `PromptBuilder.build_turn_prompt` 注入长期检索结果 + 短期摘要。
        - `PolicyClient.generate`（vLLM/FSDP 等）返回 `GenerationOutput`。
        - `StepTrajectory.from_generation` 收集 token/logprob/mask。
        - `MemoryManager.update_memory` 用生成结果更新记忆。
   - Episode 结束后用 `RewardFn` 计算 episode 级 scalar reward（用于策略），写进 `EpisodeTrajectory`；逐步奖励留给记忆 agent 后处理。

5. **TrajectoryBatchBuilder.build**
   - 仅将 `turn_role=target` 的 `StepTrajectory` 纳入训练张量；历史轮可过滤或置 `response_mask=0`。
   - 展平并 pad 到 `[num_steps, max_seq_len]`，将 episode reward 广播到目标轮 steps。
   - 额外返回 `step_meta`/`step_rewards_ext`：
     - `step_meta`：与 step 对齐的字典，含 `turn_id/turn_role/memory_ops` 等。
     - `step_rewards_ext`：预留逐步奖励占位（初始 0/None），供记忆 agent 更新，不进入 PPO 反向。
   - 输出 `PackedBatch`（`input_ids`, `logprobs`, `response_mask`, `rewards`, `group_ids`, `episode_ids`, ...），以及 sidecar 元数据。

6. **封装为 DataProto 并执行优化**
   - 将 `PackedBatch` + 需要的非 tensor 元数据（uid、原始文本、step_meta/step_rewards_ext）包装成 `DataProto`（sidecar 字段）。
   - 交给 `core_algo.update`（GRPO/PPO）完成 actor/critic/ref 的反向更新，与标准 Verl 训练流程一致。

---

## 接入提示

- **替换点**：`RayPPOTrainer.fit` 中 `gen_batch_output = self.actor_rollout_wg.generate_sequences(...)` 那一段改为调用 Phase0 rollout，并把返回的 `PackedBatch` 填充到 `batch.batch` 的相应字段。
- **Group/Reward 对齐**：`group_id` 需与 dataloader 样本分组一致，用于 GRPO 的组内归一化；reward 是 episode 级别的 scalar，广播给该 episode 所有 step。
- **多轮历史与目标轮解耦**：历史轮仅做记忆处理与压缩，目标轮才生成并进入训练；Memory/PPO 两条链路通过 `step_meta`+sidecar 解耦。
- **逐步奖励接口**：`TrajectoryBatchBuilder` 需保留 `step_meta/step_rewards_ext`，以便后续对事实拆分、长期记忆写入、短期压缩做 step-wise 奖励。
- **扩展性**：Prompt 模板、记忆写入、Policy 后端都通过接口注入，后续 Phase1/2 只需替换对应实现，RayPPOTrainer 的主流程无需再次调整。

---

（最后更新：2025-12-01）

