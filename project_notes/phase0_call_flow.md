# Phase 0 调用流（RayPPOTrainer → 自研多轮 Rollout）

> 目标：记录我们如何在 `RayPPOTrainer.fit()` 内接入 Phase 0 的多轮 rollout，并明确数据在各模块之间的流动。该文档会随着实现推进及时更新。

---

## 总览

```
RayPPOTrainer.fit
  ├─ dataloader → DialogueBatch（符合统一规范的样本）
  ├─ adapter → List[BatchInput]
  ├─ CustomRolloutWorker.run_batch(episodes)
  │     ├─ EpisodeRunner.run_episode()  # 多轮 obs→prompt→model→memory
  │     └─ TrajectoryBatchBuilder.build()  # pad & reward broadcast
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
     - `turns: List[TurnSpec]`（每轮 observation + action type）
     - `final_query`、`target_answer`、自定义 `metadata`
   - 这一步保证所有 benchmark 都喂给 rollout 相同的结构。

3. **CustomRolloutWorker.run_batch(batch_inputs)**
   - 依次调用 `EpisodeRunner.run_batch_inputs`，返回 `EpisodeTrajectory` 列表。
   - `EpisodeRunner` 会在每个 turn 将仍未完成的 `batch_inputs` 聚合成一个批次，直接把 prompt 编码成 `DataProto`，调用 Verl 的 `actor_rollout_wg.generate_sequences` 和 `compute_log_prob`，不再额外封装推理客户端。
   - 作用类似 Mem-alpha 的 `MemoryGenerationManager.run_memory_loop`，但拆成模块化组件。

4. **EpisodeRunner.run_episode**
   - 循环所有 turn：
     1. `PromptBuilder.build_turn_prompt` 生成 `PromptProto`。
     2. `PolicyClient.generate`（可接 vLLM、FSDP 等）返回 `GenerationOutput`。
     3. `StepTrajectory.from_generation` 收集 token / logprob / mask。
     4. `MemoryManager.update_memory` 用模型输出更新记忆状态。
   - Episode 结束后用 `RewardFn` 计算 scalar reward，写进 `EpisodeTrajectory`.

5. **TrajectoryBatchBuilder.build**
   - 展平所有 `StepTrajectory`，pad 到 `[num_steps, max_seq_len]`，并把 episode reward 按 step 广播。
   - 输出 `PackedBatch`（`input_ids`, `logprobs`, `response_mask`, `rewards`, `group_ids`, `episode_ids`, ...）。

6. **封装为 DataProto 并执行优化**
   - 将 `PackedBatch` + 需要的非 tensor 元数据（uid、原始文本等）包装成 `DataProto`。
   - 交给 `core_algo.update`（GRPO/PPO）完成 actor/critic/ref 的反向更新，与标准 Verl 训练流程一致。

---

## 接入提示

- **替换点**：`RayPPOTrainer.fit` 中 `gen_batch_output = self.actor_rollout_wg.generate_sequences(...)` 那一段改为调用 Phase0 rollout，并把返回的 `PackedBatch` 填充到 `batch.batch` 的相应字段。
- **Group/Reward 对齐**：`group_id` 需与 dataloader 样本分组一致，用于 GRPO 的组内归一化；reward 是 episode 级别的 scalar，广播给该 episode 所有 step。
- **扩展性**：Prompt 模板、记忆写入、Policy 后端都通过接口注入，后续 Phase1/2 只需替换对应实现，RayPPOTrainer 的主流程无需再次调整。

---

（最后更新：2025-12-01）

