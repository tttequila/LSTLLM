# GRPO 组内采样记录（VeRL版）

## 核心流程
1. **组内多采样**
   - 在 `RayPPOTrainer.fit()` 中，通过 `gen_batch.repeat(repeat_times=n, interleave=True)` 对每条输入复制 `n` 份（`@verl/verl/trainer/ppo/ray_trainer.py:1044-1047`）。
   - 复制后的 `DataProto` 交给 rollout worker (`actor_rollout_wg.generate_sequences`) 做推理，生成 `B × n` 条候选轨迹。
   - 同一原始样本的所有副本共享同一个 `group_id`（通常来自 `non_tensor_batch["uid"]`），用于后续 GRPO 归一化。

2. **轨迹回传与奖励**
   - Rollout 结束后，框架会补充 `old_log_probs`、`token_level_scores`、`token_level_rewards` 等字段，仍以 `DataProto` 形式返回。

3. **优势计算（GRPO）**
   - 在 `compute_advantage` 中，当 `config.algorithm.adv_estimator == AdvantageEstimator.GRPO` 时，调用 `core_algos.compute_grpo_outcome_advantage`（`@verl/verl/trainer/ppo/ray_trainer.py:200-275`）：
     ```python
     advantages, returns = core_algos.compute_grpo_outcome_advantage(
         token_level_rewards=data.batch["token_level_rewards"],
         response_mask=grpo_mask,
         index=data.non_tensor_batch["uid"],
         norm_adv_by_std_in_grpo=...
     )
     ```
   - 该函数按 `index`（group id）计算 `R_i - mean(R_group)` 等相对优势，并写回 `advantages`、`returns`。

4. **策略/价值更新**
   - Advantage 会被 RPC 到 `actor_rollout_wg.update_actor`、`critic_wg.update_critic` 等 worker 中，后续流程与 PPO 一致。

## 对我们项目的含义
- 只要在进入 rollout 前把 `BatchInput` 重复 `n` 次，并保持 `group_id`/`uid` 一致，标准 VeRL 就会自动执行 GRPO 组内采样和优势计算。
- Memory 操作发生在 rollout 内部（EpisodeRunner/MemoryGenerationManager），与 GRPO 的分组逻辑互不干扰。
- 如果需要控制采样次数，只需调整 `config.actor_rollout_ref.rollout.n` 或在 dataloader 层手动 repeat，无需重写 GRPO pipeline。

