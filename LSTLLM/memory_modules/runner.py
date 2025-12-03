"""负责逐输入样本的多轮环境-模型循环."""

from __future__ import annotations

from typing import Callable

import torch
from verl import DataProto

from .memory import MemoryManager, MemoryState
from .prompting import PromptBuilder
from .schemas import BatchInput, EpisodeTrajectory, GenerationOutput, PromptProto, StepTrajectory, TurnSpec

RewardFn = Callable[[BatchInput, MemoryState], float]


def default_reward_fn(batch_input: BatchInput, final_memory: MemoryState) -> float:
    """最简奖励：若存在目标答案且被最后一次摘要覆盖则给1分."""

    target = (batch_input.target_answer or "").strip().lower()
    if not target:
        return 0.0
    summaries = " ".join(str(v).lower() for v in final_memory.values())
    return 1.0 if target in summaries else 0.0


def dummy_reward_fn(batch_input: BatchInput) -> float:
    """Only for testing feasibility of the pipeline"""
    return 0.0


class EpisodeRunner:
    """执行loop的核心循环：obs→prompt→model→memory."""

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        prompt_builder: PromptBuilder,
        memory_manager: MemoryManager,
        reward_fn: RewardFn = default_reward_fn,
        device: str | torch.device = "cuda",
    ) -> None:
        self._tokenizer = tokenizer
        self._actor_rollout_wg = actor_rollout_wg
        self._prompt_builder = prompt_builder
        self._memory_manager = memory_manager
        self._reward_fn = reward_fn
        self._device = torch.device(device)

    def run_multi_turn_generation(self, batch_input: BatchInput) -> EpisodeTrajectory:
        """执行单个 batch_input，返回轨迹."""
        return self.run_multi_turn_generations([batch_input])[0]

    def run_multi_turn_generations(self, batch_inputs: list[BatchInput]) -> list[EpisodeTrajectory]:
        """批量执行多个 batch_input，并在同一 turn 内聚合 LLM 调用."""

        if not batch_inputs:
            return []

        # initialize memory states for each batch input
        memory_states = [self._memory_manager.initialize_input(bi) for bi in batch_inputs]
        
        # initialize rolling trajectories for each batch input
        trajectories = [
            EpisodeTrajectory(
                episode_id=batch_input.sample_id,
                group_id=batch_input.group_id,
                steps=[],
                reward=0.0,
                metadata={"final_query": batch_input.final_query},
            )
            for batch_input in batch_inputs
        ]
        turn_indices = [0 for _ in batch_inputs]
        # keep track of active batch inputs
        active = {idx for idx, bi in enumerate(batch_inputs) if bi.turns}

        # while still unfinished data instances in current batch 
        while active:
            batch_configs: list[PromptProto] = []
            batch_turns: list[TurnSpec] = []
            batch_input_idx: list[int] = []

            finished = []
            # iterate over still active data instances
            for bi_idx in active:
                # grep current batch input
                batch_input = batch_inputs[bi_idx]
                # skip turns that should be filtered
                while turn_indices[bi_idx] < len(batch_input.turns) and self._should_skip_turn(
                    batch_input.turns[turn_indices[bi_idx]]
                ):
                    turn_indices[bi_idx] += 1
                    
                # termination checking
                if turn_indices[bi_idx] >= len(batch_input.turns):
                    finished.append(bi_idx)
                    continue

                # grep current turn for all active data instances
                turn = batch_input.turns[turn_indices[bi_idx]]
                # render prompt
                call_config = self._prompt_builder.build_turn_prompt(batch_input, turn, memory_states[bi_idx])
                # form batch
                batch_configs.append(call_config)
                batch_turns.append(turn)
                batch_input_idx.append(bi_idx)

            # remove finished data instances
            for done_idx in finished:
                active.discard(done_idx)

            if not batch_configs:
                break

            # generate batch response by calling the actor rollout worker
            generations = self._generate_batch_single_turn(batch_configs)
            
            # parse the generations
            for generation, bi_idx, turn in zip(generations, batch_input_idx, batch_turns):
                # record the step metadata
                step_meta = self._build_step_metadata(turn, generation)
                # rolling update the trajectory
                trajectories[bi_idx].extend(StepTrajectory.from_generation(generation, metadata=step_meta))
                # update the memory state
                memory_states[bi_idx] = self._memory_manager.update_memory(memory_states[bi_idx], turn, generation.text)
                # update the turn index
                turn_indices[bi_idx] += 1
                # termination checking
                if turn_indices[bi_idx] >= len(batch_inputs[bi_idx].turns):
                    active.discard(bi_idx)

        # compute the reward and final memory state
        for idx, trajectory in enumerate(trajectories):
            trajectory.reward = self._reward_fn(batch_inputs[idx], memory_states[idx])
            trajectory.metadata["final_memory"] = memory_states[idx]
        return trajectories

    def _should_skip_turn(self, turn: TurnSpec) -> bool:
        """留钩子：可在此实现过滤逻辑."""

        return False

    def _build_step_metadata(self, turn: TurnSpec, generation: GenerationOutput) -> dict:
        """组合方便debug的元信息."""

        return {
            "turn_id": turn.turn_id,
            "action_type": turn.expected_action_type,
            "raw_text": generation.text,
            "turn_metadata": turn.metadata,
        }

    # --- Verl integration helpers -------------------------------------------------

    def _generate_batch_single_turn(self, prompt_protos: list[PromptProto]) -> list[GenerationOutput]:
        """批量生成，每个config对应一次模型调用."""
        # build the prompt batch for model (render should be done in **PromptBuilder**)
        prompt_proto = self._build_prompt_batch(prompt_protos)
        # generate response & log prob batch by calling the actor rollout worker
        gen_output_proto = self._actor_rollout_wg.generate_sequences(prompt_proto)
        logprob_proto = self._actor_rollout_wg.compute_log_prob(gen_output_proto)
        # parse the generations and return
        return self._proto_to_generations(gen_output_proto, logprob_proto, prompt_protos)

    def _build_prompt_batch(self, prompt_protos: list[PromptProto]) -> DataProto:
        """build the prompt batch for the actor rollout worker"""
        # tokenize the prompts and other necessary information for batch generation
        prompts = [proto.prompt for proto in prompt_protos]
        tokenized = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)
        position_ids = self._build_position_ids(attention_mask)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        
        # assert all sampling parameters are homogeneous
        self._assert_sampling_homogeneity(prompt_protos)
        
        ref_prompt_proto = prompt_protos[0]
        # build the meta info for the data proto
        meta_info = {
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
            "do_sample": ref_prompt_proto.temperature > 0,
            "temperature": ref_prompt_proto.temperature,
            "top_p": ref_prompt_proto.top_p,
            "recompute_log_prob": False,
            "max_new_tokens": ref_prompt_proto.max_new_tokens,
        }
        if ref_prompt_proto.stop:
            meta_info["stop"] = list(ref_prompt_proto.stop)

        # build the data proto and assign meta-info
        data_proto = DataProto.from_dict(batch)
        data_proto.meta_info = meta_info
        return data_proto

    def _proto_to_generations(
        self,
        gen_proto: DataProto,
        logprob_proto: DataProto,
        prompt_protos: list[PromptProto],
    ) -> list[GenerationOutput]:
        """将actor返回的DataProto整理为本地统一结构，并滚动更新至GenerationOutput中.
        
        Args:
            gen_proto: 标准verl生成流程返回的data proto
            logprob_proto: 标准verl生成流程返回的log prob proto
            prompt_protos: 当前batch的prompt protos
        Returns:
            generations: 更新完batch内当前turn的generation outputs
        """

        seq = gen_proto.batch["input_ids"]
        attention_mask = gen_proto.batch["attention_mask"]
        position_ids = gen_proto.batch["position_ids"]
        responses = gen_proto.batch["responses"]
        logprobs = logprob_proto.batch["old_log_probs"]

        generations: list[GenerationOutput] = []
        pad_token_id = self._tokenizer.pad_token_id or 0

        for idx in range(seq.size(0)):
            full_seq = seq[idx]
            attn = attention_mask[idx]
            pos = position_ids[idx]
            resp = responses[idx]
            lp = logprobs[idx]

            seq_len = full_seq.shape[0]
            resp_len = resp.shape[0]
            prompt_len = seq_len - resp_len

            resp_valid_mask = resp != pad_token_id
            response_mask = torch.zeros(seq_len, dtype=torch.bool, device=full_seq.device)
            response_mask[prompt_len : prompt_len + resp_len] = resp_valid_mask

            decoded = self._tokenizer.decode(resp[resp_valid_mask].tolist(), skip_special_tokens=True)
            cfg = prompt_protos[idx]

            generations.append(
                GenerationOutput(
                    input_ids=full_seq.detach().cpu(),
                    attention_mask=attn.detach().cpu(),
                    position_ids=pos.detach().cpu(),
                    logprobs=lp.detach().cpu(),
                    response_mask=response_mask.detach().cpu(),
                    text=decoded,
                    metadata={
                        "max_new_tokens": cfg.max_new_tokens,
                        "temperature": cfg.temperature,
                    },
                )
            )

        return generations

    def _build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """为prompt构造简单的递增position id."""

        seq_len = attention_mask.shape[-1]
        base = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
        return base.expand_as(attention_mask).to(torch.long)

    def _assert_sampling_homogeneity(self, prompt_protos: list[PromptProto]) -> None:
        """目前batch内要求采样参数保持一致."""

        if not prompt_protos:
            return
        ref = prompt_protos[0]
        fields = ["temperature", "top_p", "max_new_tokens", "stop"]
        for field in fields:
            ref_val = getattr(ref, field)
            for cfg in prompt_protos[1:]:
                if getattr(cfg, field) != ref_val:
                    raise ValueError(
                        f"Batching requires uniform '{field}'. Got {ref_val!r} vs {getattr(cfg, field)!r}."
                    )


