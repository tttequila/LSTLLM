"""构造多轮prompt的策略."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .memory import MemoryState
from .schemas import BatchInput, PromptProto, TurnSpec


class PromptBuilder(Protocol):
    """不同benchmark可自定义prompt模板."""

    def build_turn_prompt(self, batch_input: BatchInput, turn: TurnSpec, memory_state: MemoryState) -> PromptProto:
        """PromptProto."""


@dataclass
class DefaultPromptBuilder:
    """Phase0最简单的prompt模板."""

    system_prefix: str = "You are a helpful memory-augmented assistant."
    max_new_tokens: int = 512

    def build_turn_prompt(self, batch_input: BatchInput, turn: TurnSpec, memory_state: MemoryState) -> PromptProto:
        """把记忆渲染为纯文本提示.
        
        Args:
            batch_input: 当前batch的输入数据
            turn: 当前turn的spec
            memory_state: 当前memory state
        Returns:
            PromptProto: 构造好的PromptProto，注意该类里面的prompt是已经渲染好的prompt
        """

        memory_lines = [f"- {k}: {v}" for k, v in sorted(memory_state.items())]
        memory_block = "\n".join(memory_lines) if memory_lines else "(empty)"
        prompt = (
            f"{self.system_prefix}\n"
            f"Sample ID: {batch_input.sample_id}\n"
            f"Turn {turn.turn_id} ({turn.expected_action_type})\n"
            f"Memory:\n{memory_block}\n"
            f"Observation:\n{turn.observation}\n"
            f"Respond with the appropriate action."
        )
        return PromptProto(prompt=prompt, max_new_tokens=self.max_new_tokens)


