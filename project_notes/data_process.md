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