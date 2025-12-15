进入 rollout 前（自定义 `pre_agent_collate_fn` + 默认 collate 后，尚未封装成 DataProto）的 batch 典型 schema（张量已 stack，非张量为 `np.array(dtype=object)`；batch 维已展开为 `[batch_size * num_agents]`）：

- 张量字段
  - `input_ids`: `torch.LongTensor [B*, seq_len]`
  - `attention_mask`: `torch.LongTensor [B*, seq_len]`
  - `position_ids`: `torch.LongTensor [B*, seq_len]`（或多模态时更高维）
  - `raw_prompt_ids`: `list`→经过默认 collate 也会变成 `np.object`（若开启 `return_raw_chat/full_prompt`，同理为非张量，默认未开）
- 非张量字段（全部 `np.array(dtype=object, shape=(B*,))`）
  - `agent_role`: 每行展开后的角色 id（如 `answer_gen/fact_split/long_mem/short_mem`）
  - `pre_agent_prompt`: 对应 agent 的 messages list（未模板化、未 tokenize）
  - `question`: 字符串（按压缩策略可能为空串）
  - `chunks`: `list[str]`（按压缩策略对非 fact_split 可能为空列表）
  - `answers`: `list[str]`（非需要的 agent 可为空列表）
  - `target_answer`: 字符串
  - `data_source`, `sub_source`: 原始来源
  - `metadata`: 原始元信息（已 JSON 解析）
  - `extra_info`: `dict`，包含原始字段（`sample_id/instance_id/question_idx/turn_roles/turn_metadata/...`），并已添加：
    - `active_agent`: 当前展开的 agent role
    - `original_sample_idx`: 在原 batch 中的索引
- 形状说明：`B* = batch_size * num_agents`；张量/非张量的第一维长度一致，满足后续 `DataProto.from_single_dict` 直接封装。