"""Phase0自定义rollout worker骨架."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .memory import MemoryManager, PassthroughMemoryManager
from .packager import TrajectoryBatchBuilder
from .prompting import DefaultPromptBuilder, PromptBuilder
from .runner import EpisodeRunner, RewardFn, default_reward_fn
from .schemas import BatchInput, EpisodeTrajectory, PackedBatch


@dataclass
class RolloutWorkerConfig:
    """Rollout worker的运行配置."""

    pad_token_id: int = 0
    max_new_tokens: int = 512
    temperature: float = 0.7


class CustomRolloutWorker:
    """对Ray/VeRL完全透明的多轮rollout实现."""

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        prompt_builder: PromptBuilder | None = None,
        memory_manager: MemoryManager | None = None,
        reward_fn: RewardFn = default_reward_fn,
        batch_builder: TrajectoryBatchBuilder | None = None,
        config: RolloutWorkerConfig | None = None,
        device: str | None = None,
    ) -> None:
        self._config = config or RolloutWorkerConfig()
        # prompt wrappers
        self._prompt_builder = prompt_builder or DefaultPromptBuilder(max_new_tokens=self._config.max_new_tokens)
        # memory instances
        self._memory_manager = memory_manager or PassthroughMemoryManager()
        # episode runner, execute a sequence of generation based on each step of multi-turn conv inputs
        self._runner = EpisodeRunner(
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            prompt_builder=self._prompt_builder,
            memory_manager=self._memory_manager,
            reward_fn=reward_fn,
            device=device or "cuda",
        )
        # batch builder, pack trajectories into a batch manner
        self._batch_builder = batch_builder or TrajectoryBatchBuilder(pad_token_id=self._config.pad_token_id)

    def run_batch(self, batch_inputs: Sequence[BatchInput]) -> Tuple[list[EpisodeTrajectory], PackedBatch]:
        """执行一批batch_input，返回轨迹与pad好的batch.
        
        
        Args:
            batch_inputs: 一批batch_input spec，理想情况是一个多轮对话，每轮包含role:user, role:assistant 的chat模板
        Returns:
            trajectories: 一批轨迹
            packed: 一批pad好的batch
            
        """
        trajectories = self._runner.run_multi_turn_generations(list(batch_inputs))
        packed = self._batch_builder.build(trajectories)
        return trajectories, packed


