"""把episode轨迹转换成VeRL需要的batch张量."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch

from .schemas import EpisodeTrajectory, PackedBatch, StepTrajectory


@dataclass
class TrajectoryBatchBuilder:
    """generation output完成后的数据后处理，完成pad和reward广播."""

    pad_token_id: int = 0

    def build(self, episodes: Iterable[EpisodeTrajectory]) -> PackedBatch:
        """将若干EpisodeTrajectory打包成统一tensor.

        Args:
            episodes: run_episode产生的一批轨迹。
        Returns:
            PackedBatch，可直接交给VeRL DataProto。
        """

        episodes = list(episodes)
        if not episodes:
            raise ValueError("empty episodes input")

        flat_steps: List[StepTrajectory] = []
        episode_ids: List[int] = []
        group_ids: List[int] = []
        rewards: List[float] = []
        group_lookup: dict[str, int] = {}

        for ep_idx, ep in enumerate(episodes):
            reward = ep.reward
            rewards.append(reward)
            group_idx = group_lookup.setdefault(ep.group_id, len(group_lookup))
            for step in ep.steps:
                flat_steps.append(step)
                episode_ids.append(ep_idx)
                group_ids.append(group_idx)

        if not flat_steps:
            raise ValueError("episodes contain zero steps")

        seq_lens = [step.input_ids.shape[0] for step in flat_steps]
        max_seq_len = max(seq_lens)
        num_steps = len(flat_steps)

        input_ids = torch.full((num_steps, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((num_steps, max_seq_len), dtype=torch.bool)
        position_ids = torch.zeros((num_steps, max_seq_len), dtype=torch.long)
        logprobs = torch.zeros((num_steps, max_seq_len), dtype=torch.float32)
        response_mask = torch.zeros((num_steps, max_seq_len), dtype=torch.bool)
        reward_tensor = torch.zeros(num_steps, dtype=torch.float32)

        for row_idx, step in enumerate(flat_steps):
            seq_len = step.input_ids.shape[0]
            input_ids[row_idx, :seq_len] = step.input_ids
            attention_mask[row_idx, :seq_len] = step.attention_mask.to(torch.bool)
            position_ids[row_idx, :seq_len] = step.position_ids
            logprobs[row_idx, :seq_len] = step.logprobs
            response_mask[row_idx, :seq_len] = step.response_mask
            reward_tensor[row_idx] = rewards[episode_ids[row_idx]]

        return PackedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logprobs=logprobs,
            response_mask=response_mask,
            rewards=reward_tensor,
            group_ids=torch.tensor(group_ids, dtype=torch.long),
            episode_ids=torch.tensor(episode_ids, dtype=torch.long),
        )


