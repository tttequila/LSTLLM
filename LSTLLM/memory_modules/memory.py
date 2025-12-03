"""Phase0记忆模块的抽象."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .schemas import BatchInput, TurnSpec

from dataclasses import dataclass

# Memory state for single data instance
# TODO: unfinished yet
class MemoryState:
    MAX_MEMORY_ITEMS = 20 # maximum number of memory items to be stored for each type of memory
    MEMORY_CONSOLIDATE_STEP = 5 # number of memory items to be consolidated at a time
    TOPK = 20 # top-k
    
    core_memory: str
    episodic_memory: List[Dict[str, str]]
    semantic_memory: List[Dict[str, str]]

class MemoryManager(Protocol):
    """统一不同记忆策略的接口。对复数个memory state进行管理"""

    def initialize_input(self, batch_input: BatchInput) -> MemoryState:
        """为新的输入样本创建初始记忆."""

    def update_memory(self, state: MemoryState, turn: TurnSpec, model_text: str) -> MemoryState:
        """根据模型输出更新记忆."""


class PassthroughMemoryManager:
    """Phase0占位实现：不做任何记忆写入."""

    def initialize_input(self, batch_input: BatchInput) -> MemoryState:
        return {"sample_metadata": batch_input.metadata}

    def update_memory(self, state: MemoryState, turn: TurnSpec, model_text: str) -> MemoryState:
        state = dict(state)
        state[f"turn_{turn.turn_id}_summary"] = model_text.strip()
        return state


