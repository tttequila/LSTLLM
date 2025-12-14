"""
Utility helpers that can be reused across the LSTLLM package.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

import logging

logger = logging.getLogger(__name__)

# 若需降低内存占用，可只在部分 agent 上保留大字段
COMPACT_NON_ACTIVE = True


def _maybe_load_json(idx: int, field: str, value: Any) -> Any:
    """
    Best-effort JSON loader used by the multi-agent collate function.

    RLHFDataset 读出的字段大多是字符串（因为前置的 parquet 写入时做了 JSON dumps）。
    这里统一尝试反序列化；若失败则直接返回原值，避免因为少数非 JSON 改写数据而报错。
    """

    if isinstance(value, str):
        striped = value.strip()
        if not striped:
            return value
        try:
            return json.loads(striped)
        except json.JSONDecodeError:
            # debug
            logger.warning(f"Failed to load JSON: {field} of {idx}-th sample; Loaded value: {value}")
            return value
    return value


def _ensure_list(obj: Any) -> List[Any]:
    """
    将 answers / chunks 等字段归一化为 list，方便后续 batch 展开。
    """

    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def pre_agent_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, List[Any]]:
    """
    多 agent collate 样例：将 `extra_info["pre_agent"]` 展平合并到 batch 维。

    设计要点：
    - RLHFDataset 输出的每一条样本可能包含多个 agent prompt（保存在 extra_info.pre_agent 内）。
    - collate 负责复制 QA 公共字段，并为每个 agent 单独产出一条记录，从而得到 `[batch * agents, ...]`。
    - 不做 tokenizer 编码 / padding，只返回 Python 对象，后续 rollout 再决定如何渲染模板。

    Args:
        batch: DataLoader 传入的样本列表（每项是 RLHFDataset.__getitem__ 的返回 dict）。

    Returns:
        Dict[str, List[Any]]: key → list，对应展开后的 batch。常见字段：
            - `agent_role`: 当前 agent 的角色 id。
            - `pre_agent_prompt`: 该 agent 的 prompt（仍为 messages list，未模板化）。
            - `question` / `chunks` / `answers` / `target_answer`: QA 级数据。
            - `extra_info`: 追加了 `active_agent` 的 dict，供 rollout 侧读取 meta。
            - `metadata`: 原始 meta 信息（去 JSON 之后）。
            - 其他字段可按需补充。
    """

    expanded: List[Dict[str, Any]] = []

    for sample_idx, sample in enumerate(batch):
        # 统一 JSON 反序列化
        chunks = _ensure_list(_maybe_load_json(sample_idx, "chunks", sample.get("chunks")))
        answers = _ensure_list(_maybe_load_json(sample_idx, "answers", sample.get("answers")))
        metadata = _maybe_load_json(sample_idx, "metadata", sample.get("metadata")) or {}

        extra_info = _maybe_load_json(sample_idx, "extra_info", sample.get("extra_info"))
        pre_agent_prompts = extra_info.get("pre_agent", None) 

        assert pre_agent_prompts is not None, f"pre_agent_prompts is None for {sample_idx}-th sample"

        num_agents = len(pre_agent_prompts)

        # take original sample, repeat num_agents times and extend to expanded list

        start = len(expanded)
        expanded.extend([sample.copy() for _ in range(num_agents)])

        for offset, (agent_role, agent_prompt) in enumerate(pre_agent_prompts.items()):
            keep_chunks = not COMPACT_NON_ACTIVE or agent_role == "fact_split"
            keep_question = not COMPACT_NON_ACTIVE or agent_role in ("fact_split", "answer_gen")

            item = expanded[start + offset]
            item["agent_role"] = agent_role
            item["pre_agent_prompt"] = agent_prompt
            item["question"] = question if keep_question else ""
            item["chunks"] = chunks if keep_chunks else []
            item["extra_info"] = {
                **extra_info,
                "active_agent": agent_role,
                "original_sample_idx": sample_idx,
            }

    # 将 list[dict] 转成 dict[list]，方便 Trainer 后续访问
    batched: Dict[str, List[Any]] = defaultdict(list)
    for flattened_sample in expanded:
        for key, value in flattened_sample.items():
            batched[key].append(value)
            
    # access default collate fn to align the standard format for verl
    batched = default_collate_fn(batched)

    return batched