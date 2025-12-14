"""
Utility helpers that can be reused across the LSTLLM package.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn  # type: ignore[import]

import logging

logger = logging.getLogger(__name__)

# 若需降低内存占用，可只在部分 agent 上保留大字段
COMPACT_NON_ACTIVE = True
# REQUIRED_JSON_FIELDS = {"extra_info", "chunks", "question", "answers"}


# def _maybe_load_json(idx: int, field: str, value: Any) -> Any:
#     """
#     stored type tolerant json loader, ensure loaded value exists and is a dict or a list
#     """

#     must_parse = field in REQUIRED_JSON_FIELDS

#     # if value is a string, try to load it as a json
#     if isinstance(value, str):
#         striped = value.strip()
#         # ensure field exists and is not empty
#         if not striped:
#             if must_parse:
#                 raise ValueError(f"Empty JSON string for required field `{field}` in sample {idx}.")
#             return value
#         try:
#             value = json.loads(striped)
#         # if failed, return original value or raise error if this field is necessary
#         except json.JSONDecodeError as exc:
#             message = f"Failed to load JSON for `{field}` of {idx}-th sample: {exc}; payload={value}"
#             if must_parse:
#                 raise ValueError(message) from exc
#             logger.warning(message)
#             return value

#     return value


def _ensure_list(name: str, obj: Any) -> List[Any]:
    """
    将 answers / chunks 等字段归一化为 list，方便后续 batch 展开。
    """

    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to load JSON as list for `{name}`, please check the data format")
    return obj


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

        extra_info = sample.get("extra_info")
        
        if not isinstance(extra_info, dict):
            raise TypeError(f"`extra_info` of sample {sample_idx} must be dict, got {type(extra_info)}")

        # pop and get pre_agent_prompts from extra_info
        pre_agent_prompts = extra_info.pop("pre_agent", None)
        assert isinstance(pre_agent_prompts, dict) and pre_agent_prompts, (
            f"pre_agent_prompts is None for {sample_idx}-th sample"
        )

        chunks_parsed = _ensure_list("chunks", sample.get("chunks"))
        question_parsed = sample.get("question", "")
        answers_parsed = _ensure_list("answers", sample.get("answers"))

        num_agents = len(pre_agent_prompts)

        # take original sample, repeat num_agents times and extend to expanded list
        start = len(expanded)
        expanded.extend([sample.copy() for _ in range(num_agents)])

        for offset, (agent_role, agent_prompt) in enumerate(pre_agent_prompts.items()):
            keep_chunks = not COMPACT_NON_ACTIVE or agent_role == "fact_split"
            keep_qna = not COMPACT_NON_ACTIVE or agent_role in ("fact_split", "answer_gen")

            item = expanded[start + offset]
            
            # add agent role and prompt to the item
            item["agent_role"] = agent_role
            item["pre_agent_prompt"] = agent_prompt

            # drop chunks and question if required to save memory
            item["chunks"] = chunks_parsed if keep_chunks else []
            item["question"] = question_parsed if keep_qna else ""
            item["answers"] = answers_parsed if keep_qna else []

            # update extra_info with dropped pre_agent_prompts
            item["extra_info"] = extra_info.copy()

    # call default collate fn to align the standard format for verl
    return default_collate_fn(expanded)