"""Phase 0 rollout框架公共接口."""

from .schemas import (
    BatchInput,
    EpisodeTrajectory,
    GenerationOutput,
    PackedBatch,
    PromptProto,
    StepTrajectory,
    TurnSpec,
)
from .prompting import DefaultPromptBuilder, PromptBuilder
from .memory import MemoryManager, PassthroughMemoryManager
from .runner import EpisodeRunner, RewardFn, default_reward_fn
from .packager import TrajectoryBatchBuilder
from .rollout_worker import CustomRolloutWorker, RolloutWorkerConfig

__all__ = [
    # Schemas
    "BatchInput",
    "TurnSpec",
    "PromptProto",
    "GenerationOutput",
    "StepTrajectory",
    "EpisodeTrajectory",
    "PackedBatch",
    # Prompting / memory
    "PromptBuilder",
    "DefaultPromptBuilder",
    "MemoryManager",
    "PassthroughMemoryManager",
    # Runner / batching
    "EpisodeRunner",
    "TrajectoryBatchBuilder",
    "RewardFn",
    "default_reward_fn",
    # Worker
    "CustomRolloutWorker",
    "RolloutWorkerConfig",
]

