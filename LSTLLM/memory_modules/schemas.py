"""多轮RL rollout需要的核心数据结构."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import torch

Tensor = torch.Tensor
MemoryState = Dict[str, Any]


@dataclass(frozen=True)
class TurnSpec:
    """描述单轮输入。

    Args:
        turn_id: turn的顺序编号。
        observation: 环境提供的纯文本观察。
        expected_action_type: 该轮模型应输出的动作类型，例如"memory_ops"或"answer"。
        turn_role: "history" 仅做记忆处理；"target" 需要生成+训练。
        metadata: 记录benchmark相关额外信息（例如chunk id、speaker等）。
    """

    turn_id: int
    observation: str
    expected_action_type: str
    turn_role: str = "target"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchInput:
    """符合统一规范的输入序列描述."""

    sample_id: str
    group_id: str
    turns: Sequence[TurnSpec]
    final_query: str
    target_answer: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptProto:
    """封装一次模型调用的prompt与采样参数."""

    prompt: str
    max_new_tokens: int = 512
    stop: Sequence[str] | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    logit_bias: Dict[int, float] | None = None
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationOutput:
    """模型一次生成返回的token级信息."""

    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    position_ids: torch.LongTensor
    logprobs: torch.FloatTensor
    response_mask: torch.BoolTensor
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepTrajectory:
    """保存单次模型调用产生的轨迹."""

    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    position_ids: torch.LongTensor
    logprobs: torch.FloatTensor
    response_mask: torch.BoolTensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_reward: float | None = None  # 预留逐步奖励，供记忆agent使用

    @classmethod
    def from_generation(cls, generation: GenerationOutput, metadata: Dict[str, Any] | None = None) -> "StepTrajectory":
        """辅助方法：由模型输出转换为StepTrajectory."""

        meta = metadata or dict(generation.metadata)
        return cls(
            input_ids=generation.input_ids,
            attention_mask=generation.attention_mask,
            position_ids=generation.position_ids,
            logprobs=generation.logprobs,
            response_mask=generation.response_mask,
            metadata=meta,
            step_reward=meta.get("step_reward") if isinstance(meta, dict) else None,
        )


@dataclass
class EpisodeTrajectory:
    """一条完整episode的轨迹. 滚动记录每轮模型调用的结果，并收集迭代."""

    episode_id: str
    group_id: str
    steps: list[StepTrajectory]
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_meta: list[Dict[str, Any]] = field(default_factory=list)  # 与step或history事件对齐的元信息
    step_rewards_ext: list[float] = field(default_factory=list)  # 逐步奖励占位（不进入PPO反向）

    def extend(self, step: StepTrajectory) -> None:
        """追加新的step，供runner逐轮写入。"""

        self.steps.append(step)


@dataclass
class PackedBatch:
    """Pad之后喂入VeRL的tensor集合."""

    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor
    position_ids: torch.LongTensor
    logprobs: torch.FloatTensor
    response_mask: torch.BoolTensor
    rewards: torch.FloatTensor
    group_ids: torch.LongTensor
    episode_ids: torch.LongTensor
    # sidecar，不参与PPO反向，但供记忆agent/调试使用
    step_meta: list[Dict[str, Any]] | None = None
    step_rewards_ext: torch.FloatTensor | None = None


