以下是最终共识的要点与实现思路总结（去除了中途被否定或不需要的内容）：

1) 多轮/多步训练的基本原则  
- 只要 `response_mask` 正确标记生成 token，多轮拼接后的序列都能在一次 PPO 更新中被训练。  
- reward 可以是全局广播，也可以分步写入对应 token 片段；`compute_advantage`/`update_actor` 按 mask 计算梯度。  
- 多轮条件已体现在生成时的 `input_ids/attention_mask` 上，log_prob/梯度自动在各自上下文下计算。

2) 并列分支、同参多角色  
- 无需把分支强行串成一条序列；把每个角色的样本作为独立样本放入同一 batch。  
- 共享一套参数/LoRA，单次 `update_actor`，梯度是所有样本的总和，不能按角色拆参或拆优化。  
- 奖励可按 `agent_id`/`group_id` 分支，写入各自 `token_level_scores`，无关样本/位置填 0。

3) 数据与采样（保持组内依赖）  
- 数据集只存静态素材和角色信息：system prompt/模板、问题、chunk、工具配置、`agent_id`、`group_id` 等，不需要预先放入上游输出。  
- 若必须让一组多agent样本同行：自定义 Dataset/Collate/Sampler，让 DataLoader 返回“组”，再在 collate 里将组维展平到 batch 维，长度对齐（padding）后送入 rollout。  
- 若不强制成组，可直接当独立样本混批，但对于有显式依赖的流水线，推荐成组。

4) Rollout（自定义，多轮/分支/LoRA/vLLM）  
- 在组内按依赖顺序生成：用上游 agent 的 response 动态拼下游 agent 的 prompt。  
- 返回 DataProto 时提供真实生成用的张量：  
  - `input_ids`、`attention_mask`（覆盖 prompt+response 全长）、`position_ids`  
  - `responses`、`response_mask`（生成=1，观测/工具/填充=0）  
  - 可选 `rollout_log_probs`（便于 bypass/rollout_correction）  
  - 非张量字段与 batch_size 对齐（`agent_id/group_id/data_source` 等）。  
- 保证 padding 对齐：`input_ids` 左填充，`responses/response_mask` 右填充；batch 内形状一致即可。  
- Trainer 的 `union` 会用生成返回的字段覆盖初始 batch，因此初始 prompt 不会造成对齐问题。

5) 奖励与优势  
- reward_fn 按需分支（agent_id/group_id），写入 `token_level_scores`；未使用的样本/位置置 0。  
- KL/ref/critic 在同一大 batch 上一次计算；mask 确保只作用在生成 token。

6) RLHFDataset 与 prompt 字段  
- 数据阶段可将 prompt 填为角色 system prompt/模板或占位；真正的完整 prompt（含上游输出、chunk等）在 rollout 中动态拼接。  
- 静态信息放在非张量字段（extra_info）便于 rollout/reward 读取。

7) Mem-alpha 的参考做法  
- 用自定义生成管理器（MemoryGenerationManager）在训练时动态构建 prompt、做多轮/记忆处理，再返回完整 DataProto；奖励自定义、多路汇总，最后仍是标准 PPO 流程单次更新。

8) vLLM+LoRA 自定义 rollout 规范  
- 继承 BaseRollout，实现 `resume/update_weights/release/generate_sequences`。  
- `generate_sequences` 返回上述标准字段；LoRA 权重通过 `update_weights`（带 peft_config/base_sync_done）加载。  
- 多轮/工具场景下 `response_mask` 要正确标 1/0，工具返回/观测置 0。

9) 维度/对齐风险  
- 只要最终返回的张量形状一致（同一批次），mask 正确，非张量字段长度等于 batch_size，Trainer 不会因“多角色/不同长度”报维度错。  
- 成组展开时注意 repeat/interleave 后的索引对齐（group_id/agent_id 与张量行对应）。

整体实现路径：  
- 数据：存静态素材+角色信息，选用自定义 Dataset/Collate 保组。  
- Rollout：组内依赖生成，动态拼 prompt，返回完整 DataProto（含 mask）。  
- 奖励：按角色/组分支写 token_level_scores。  
- 训练：保持标准 RayPPOTrainer 流程（单次 update，单套参数），利用 mask 让多轮/多分支样本共同更新。