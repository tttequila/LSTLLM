"""
MemoryAgentBench 数据清洗：将 chunk-level 数据展开为 QA-level，
输出符合 RLHFDataset 约定的 `prompt` 与 `extra_info` 列。

注意：不依赖外部 LLM API，仅做结构转换与简单分块。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Optional

import pandas as pd
from datasets import load_dataset, concatenate_datasets
import tiktoken
import random
from transformers import AutoTokenizer


# UNI_SYS_PROMPTS = []

FACT_SPLIT_SYS_PROMPTS = [
    "You are a precision-oriented analytical agent. When given input, isolate all distinct factual statements, ensuring each is atomic, explicit, and independently meaningful. Avoid inference unless it is strictly entailed.",
    "Your task is to extract granular facts from provided content. Represent each fact as a standalone assertion and assess its relevance strictly with respect to the current query context.",
    "Process information by decomposing it into minimal factual units. Distinguish core statements from supporting details and tag each unit based on its situational significance.",
    "Analyze the input to identify discrete, verifiable facts. Separate compound statements and evaluate each resulting fact against the active question or objective.",
    "Operate with a bias toward over-segmentation rather than aggregation. Each fact should contain only one subject-predicate relationship and be evaluated for contextual importance.",
    "Treat the provided content as a source of claims. Enumerate those claims individually and determine whether each contributes meaningfully to solving the present task.",
    "Extract factual elements without summarization. Preserve original meaning while ensuring that each fact is isolated, concise, and relevance-scored.",
    "Your output should reflect a structured breakdown of factual content. Avoid narrative flow; focus instead on clarity, separation, and relevance judgment.",
    "Identify explicit and implicit factual statements, normalize them into clear assertions, and determine their priority relative to the user's current intent.",
    "You function as a factual parser. Decompose inputs into independent facts and classify their utility for downstream reasoning in the current context."
]

LONG_MEM_SYS_PROMPTS = [
    "You are responsible for curating durable knowledge assets. Evaluate incoming information for enduring value, stability over time, and likelihood of reuse across future interactions.",
    "Manage information with an emphasis on persistence and reuse. Retain only content that is broadly applicable, identity-defining, or strategically valuable beyond the immediate context.",
    "Your role is to organize and maintain structured memory entries. Prioritize clarity, deduplication, and semantic consistency across stored knowledge.",
    "Assess whether information represents a long-lived preference, constraint, capability, or factual constant. If so, refine and store it in an organized form.",
    "Operate conservatively: only commit information that is reliable, non-ephemeral, and potentially useful in multiple future scenarios.",
    "Continuously maintain a clean and coherent memory base. Merge overlapping entries, resolve conflicts, and discard data that has become obsolete or redundant.",
    "Transform qualifying information into normalized memory records that are easy to retrieve, interpret, and apply in varied contexts.",
    "Focus on long-horizon utility. Favor stable facts, enduring user attributes, and recurring patterns over situational or transient details.",
    "Apply governance to stored knowledge. Ensure each retained item has clear scope, justification, and alignment with long-term usefulness.",
    "You act as a steward of persistent knowledge. Optimize for accuracy, longevity, and minimal noise in retained memory."
]

SHORT_MEM_SYS_PROMPTS = [
    "You manage transient contextual information. Compress, summarize, and adapt recent inputs to support the current interaction without preserving unnecessary detail.",
    "Your function is to maintain a concise working context. Integrate recent facts with existing state and reduce them to their most actionable form.",
    "Operate with awareness of limited capacity. Prioritize relevance, recency, and task alignment when shaping short-term memory representations.",
    "Transform raw, low-priority facts into compact summaries that preserve intent and context while minimizing verbosity.",
    "Balance completeness with efficiency. Retain just enough information to support immediate reasoning and discard excess detail.",
    "Continuously update the active context as new information arrives. Re-evaluate what should remain salient and what can be abstracted away.",
    "Your outputs should reflect synthesized context rather than raw data. Emphasize coherence and usability for downstream response generation.",
    "Treat short-term memory as fluid. Adapt summaries dynamically as goals shift or new constraints emerge.",
    "Focus on integration rather than storage. Ensure that recent information aligns with the current conversational or task state.",
    "You serve as a context optimizer. Shape temporary memory to maximize relevance and minimize cognitive overhead."
]

ANSWER_SYS_PROMPTS = [
    "You are an assistant focused on delivering useful, accurate, and context-aware responses by leveraging all available memory and current input.",
    "Generate outputs that directly address the user's needs, applying stored knowledge and contextual understanding in a coherent and practical manner.",
    "Your priority is user value. Synthesize relevant information into clear, actionable, and well-structured responses.",
    "Operate as a reasoning and communication agent. Use provided context and memory to inform answers without exposing internal processes.",
    "Adapt tone, depth, and format to suit the user's intent while maintaining correctness and clarity.",
    "When generating responses, integrate persistent knowledge with situational context to produce results that are both informed and relevant.",
    "Resolve ambiguity where possible and make reasonable assumptions explicit only when necessary to move the interaction forward.",
    "Focus on outcomes rather than mechanisms. Deliver answers, explanations, or generated content that effectively satisfy the request.",
    "Ensure internal consistency across responses by aligning with previously established context and retained knowledge.",
    "You function as the primary interface to the user. Convert memory and reasoning into helpful, intelligible output."
]

# -------------------------
# 辅助工具
# -------------------------

def _ensure_parent_dir(path: str) -> None:
    """创建输出文件父目录。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _save_rows(rows: Sequence[Dict[str, Any]], output_path: str) -> str:
    """将样本列表保存为 parquet，并返回最终路径。"""
    _ensure_parent_dir(output_path)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    return output_path


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding:
    """缓存 tiktoken 编码器，默认使用 cl100k_base。"""
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """基于 tiktoken 的 token 计数，失败时降级为空格计数。"""
    try:
        return len(_get_encoding().encode(text))
    except Exception:
        return len(text.split())


def _sent_tokenize(text: str) -> List[str]:
    """句子级切分，优先使用 nltk，缺失时回退正则。"""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        return [s for s in re.split(r"(?<=[。！？!?\.])\s+", text) if s.strip()]


def _chunk_by_sentences(text: str, max_tokens: int) -> List[str]:
    """
    对齐 mem-alpha 的句子级分块：按句子累积到 token 上限，不再做词级拆分。
    """
    if not text:
        return []

    sentences: List[str] = _sent_tokenize(text)

    chunks: List[str] = []
    current = ""
    current_tokens = 0

    for sent in sentences:
        if "<|endoftext|>" in sent:
            sent = sent.replace("<|endoftext|>", "\n")

        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(current.strip())
            current = sent
            current_tokens = sent_tokens
        else:
            if current:
                current = f"{current} {sent}"
                current_tokens += sent_tokens + _count_tokens(" ")
            else:
                current = sent
                current_tokens = sent_tokens

    if current:
        chunks.append(current.strip())

    return chunks


def _build_prompt(agent_id: Optional[str] = None) -> List[Dict[str, str]]:
    """生成 RLHF 期望的 prompt 结构（系统 + 最终 user 提问）。"""
    
    # # sample from uniform prompts
    # if agent_id == None:
    #     return [
    #         {"role": "system", "content": random.choice(UNI_SYS_PROMPTS)},
    #         # {"role": "user", "content": f"{agent_id}: {question}"},
    #     ]
    # sample from according agent prompt pools
    if agent_id == 'fact_split':
        return [
            {"role": "system", "content": random.choice(FACT_SPLIT_SYS_PROMPTS)},
            # {"role": "user", "content": f"{agent_id}: {question}"},
        ]
    elif agent_id == 'long_mem':
        return [
            {"role": "system", "content": random.choice(LONG_MEM_SYS_PROMPTS)},
            # {"role": "user", "content": f"{agent_id}: {question}"},
        ]
    elif agent_id == 'short_mem':
        return [
            {"role": "system", "content": random.choice(SHORT_MEM_SYS_PROMPTS)},
            # {"role": "user", "content": f"{agent_id}: {question}"},
        ]
    elif agent_id == 'answer_gen':
        return [
            {"role": "system", "content": random.choice(ANSWER_SYS_PROMPTS)},
            # {"role": "user", "content": f"{agent_id}: {question}"},
        ]
    else:
        raise ValueError(f"Invalid agent role: {agent_id}")

def _qa_rows_from_entry_multi_agent(
    *,
    chunks: List[str],
    qa_pairs: Sequence[Dict[str, Any]],
    base_sample_id: str,
    data_source: str,
    sub_source: str | None,
    agent_ids: Sequence[str] = ["answer_gen", "fact_split", "long_mem", "short_mem"],
    reading_dates: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    在 QA 样本上附加 agent 维度信息，便于后续在 collate 阶段将 agent 维展开到 batch。
    """
    
    rows = []

    for q_idx, qa in enumerate(qa_pairs):
        
        question = qa.get("question") or ""
        answers = qa.get("answer") or qa.get("answers") or ""
        # possible multiple answers
        answers_list = answers if isinstance(answers, list) else [answers]

        # build a prompt to hold prompt key in data instance (required by RLHFDataset for an existing prompt key)
        prompt = _build_prompt("answer_gen")

        # meta-info
        sample_id = str(base_sample_id)
        instance_id = f"{sample_id}-q{q_idx}"

        # chunk related
        turn_roles = ["history"] * len(chunks) + ["target"]
        turn_metadata = [{} for _ in chunks]

        extra_info: Dict[str, Any] = {
            "question": question,
            "answers": answers_list,
            "target_answer": answers_list[0] if answers_list else "",
            "chunks": chunks,
            "data_source": data_source,
            "sub_source": sub_source,
            "agent_role": "answer_gen",
            "pre_agent": {
                agent_id: _build_prompt(agent_id) for agent_id in agent_ids
                },
            "meta_info":{
                "sample_id": sample_id,
                "instance_id": instance_id,
                "question_idx": q_idx,
                "turn_roles": turn_roles,
                "turn_metadata": turn_metadata,
            }
        }
        if reading_dates:
            extra_info["reading_dates"] = reading_dates

        rows.append(
            {
                "prompt": prompt,
                "extra_info": extra_info,
            }
        )
        
    return rows

    
def count_tokens_qwen(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(text))

# -------------------------
# 处理路径 1：直接读取 HF raw 数据
# -------------------------

def _chunk_context_longmemeval(contexts: str, max_tokens: int, tokenizer: AutoTokenizer) -> List[str]:
    """
    将longmemeval中，单条数据样本里的多轮对话上下文拆分成多个chunks，每个chunk包含一个时间戳和一段对话
    
    Args:
        contexts: 原始context字段，为字符串，需要eval或json.loads转换为list
        max_tokens: 每个chunk的最大token数
        tokenizer: 分词器
        
    Returns:
        chunks: List[str]，每个元素为一段chunk，每个chunk包含一个时间戳和一段对话
    """
    try:
        all_context = eval(contexts) if isinstance(contexts, str) else contexts
    except Exception:
        all_context = json.loads(contexts)
    
    assert isinstance(all_context, list), "Expected all_context to be a list"
    assert len(all_context) % 2 == 0, "Expected alternating [timestamp, session] pairs"
    
    for idx in range(0, len(all_context), 2):
        ts = all_context[idx]
        session = all_context[idx + 1]
        assert isinstance(ts, str), f"Expected timestamp at index {idx} to be str"
        assert isinstance(session, list), f"Expected session at index {idx+1} to be list"
        for turn in session:
            assert isinstance(turn, dict) and 'role' in turn and 'content' in turn, "Invalid turn format"

                        
        def render_session(timestamp_str, turns, continuation=False):
            """将timestamp和turns渲染成带时间戳的对话轮次"""
            header = f"[Dialogue at timestamp {timestamp_str}]"
            turns_text = "\n".join(
                f"<{turn['role'].capitalize()}>{turn['content']}" for turn in turns
            )
            return f"{header}\n{turns_text}"
            
        def split_large_turn(turn, max_tokens=2048):
            """将对话拆分成多个chunks，每个chunk包含一个角色和一段对话"""
            content = turn['content']
            role = turn['role']

            # 如果对话轮次足够小，则直接返回
            test_turn = {'role': role, 'content': content}
            if count_tokens_qwen(f"<{role.capitalize()}>{content}", tokenizer) <= max_tokens - 150:  # Leave some buffer for headers
                return [test_turn]
            import re
            # 先尝试按句子拆分
            sentences = re.split(r'(?<=[.!?])\s+', content)

            split_turns = []
            current_content = ""

            # 遍历每个句子，将句子拆分成多个chunks
            for sentence in sentences:
                splited_content  = current_content + (" " if current_content else "") + sentence
                splited_text = f"<{role.capitalize()}>{splited_content}"
                # 如果句子长度足够小，则直接加入当前chunk
                if count_tokens_qwen(splited_text, tokenizer) <= max_tokens - 150:
                    current_content = splited_content
                else:
                    if current_content:
                        split_turns.append({'role': role, 'content': current_content.strip()})
                        current_content = sentence
                    else:
                        words = sentence.split()
                        chunk = ""
                        for word in words:
                            test_chunk = chunk + (" " if chunk else "") + word
                            if count_tokens_qwen(f"<{role.capitalize()}>{test_chunk}", tokenizer) <= max_tokens - 150:
                                chunk = test_chunk
                            else:
                                if chunk:
                                    split_turns.append({'role': role, 'content': chunk.strip()})
                                chunk = word
                        
                        if chunk:
                            current_content = chunk

            if current_content:
                split_turns.append({'role': role, 'content': current_content.strip()})

            return split_turns

        def split_session_into_segments(timestamp_str, turns):
            """将对话轮次拆分成多个segments，每个segment包含一个时间戳和一段对话"""
            segments = []
            current_turns = []

            # 遍历每个对话轮次，将每个对话轮次拆分成多个chunks
            for turn in turns:
                # 检查单个对话轮次是否超过最大长度
                single_turn_text = f"<{turn['role'].capitalize()}>{turn['content']}"
                # 如果单个对话轮次超过最大长度，则将对话轮次拆分成多个chunks
                if count_tokens_qwen(single_turn_text, tokenizer) > 2048 - 150:  # Leave buffer for headers
                    # 将对话轮次拆分成多个chunks，返回轮次列表
                    split_turns = split_large_turn(turn, max_tokens=2048)

                    # 遍历每个拆分后的对话轮次，将每个对话轮次添加到当前segments
                    for split_turn in split_turns:
                        
                        test_turns = current_turns + [split_turn]
                        is_cont = len(segments) > 0
                        candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)
                        if count_tokens_qwen(candidate_text, tokenizer) <= 2048:
                            current_turns = test_turns
                        else:
                            if current_turns:
                                if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                    last_turn = current_turns.pop()
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [last_turn, split_turn]
                                else:
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [split_turn]
                            else:
                                current_turns = [split_turn]
                                
                else:
                    test_turns = current_turns + [turn]
                    is_cont = len(segments) > 0
                    candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)
                    
                    if count_tokens_qwen(candidate_text, tokenizer) <= 2048:
                        current_turns = test_turns
                    else:
                        if current_turns:
                            if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                last_turn = current_turns.pop()
                                segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                current_turns = [last_turn, turn]
                            else:
                                segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                current_turns = [turn]
                        else:
                            segments.append(render_session(timestamp_str, [turn], continuation=is_cont))
                            current_turns = []

            if current_turns:
                segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))

            return segments

        chunks = []
        current_chunk = ""
        for idx in range(0, len(all_context), 2):
            ts = all_context[idx]
            session = all_context[idx + 1]
            session_segments = split_session_into_segments(ts, session)
            for segment_text in session_segments:
                if current_chunk:
                    candidate = current_chunk + "\n\n" + segment_text
                else:
                    candidate = segment_text
                    
                candidate_tokens = count_tokens_qwen(candidate, tokenizer)

                if candidate_tokens > 2048:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = segment_text
                elif candidate_tokens >= 1800:
                    chunks.append(candidate)
                    current_chunk = ""
                else:
                    current_chunk = candidate

        if current_chunk:
            lines = current_chunk.strip().split('\n')
            if lines and lines[-1].startswith('<User>'):
                pass
            chunks.append(current_chunk)

        for chunk in chunks:
            assert count_tokens_qwen(chunk, tokenizer) <= 2048

        return chunks
    
def process_memoryagentbench_raw(
    *,
    split: Optional[str] = None,
    output_path: str = "./data/memoryagentbench_qa_raw.parquet",
    max_tokens_per_chunk: int = 2048,
    keywords_path: Optional[str] = None,
) -> str:
    """
    从 HuggingFace 原始数据集生成 QA-level parquet。

    仅做轻量切分（按 token 数），方便后续 replay chunk。
    """
    if not split:
        ds = load_dataset("ai-hyz/MemoryAgentBench")
        split = list(ds.keys())
        ds = [ds[split_name] for split_name in split]
    else:
        ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)
        split = [split]
        ds = [ds]
    
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
    
    rows: List[Dict[str, Any]] = []
    
    if 'Long_Range_Understanding' in split:
        if keywords_path is not None:
            with open(keywords_path, 'r') as file:
                answers_to_keywords_mapping = json.load(file)
        else:
            raise ValueError("keywords_path is required for Long_Range_Understanding split")

    for split_name, dataset in zip(split, ds):
    # iterate over each item in the dataset
        for item_idx, item in enumerate(dataset):
            
            source = item['metadata']['source']
            context = item['context']
            questions = item['questions']
            answers = item['answers']
            metadata = item['metadata']
            
            # filter out some data sources
            if source in ['recsys_redial_full']:
                continue

            # special chunk field, list of chunks, each chunk contains a timestamp and a session of turns
            if source == 'longmemeval_s*':
                chunks = _chunk_context_longmemeval(context, max_tokens_per_chunk, qwen_tokenizer)
                qa_pairs = [{'question': question, 'answer': answer} for question, answer in zip(questions, answers)]
            
            elif split == 'Test_Time_Learning':
                chunks = _chunk_by_sentences(context, 1024)
                qa_pairs = [{'question': question, 'answer': answer} for question, answer in zip(questions, answers)]

            elif split_name == 'Accurate_Retrieval':
                chunks = _chunk_by_sentences(context, 2048)
                qa_pairs = [{'question': question, 'answer': answer} for question, answer in zip(questions, answers)]
            else:
                chunks = _chunk_by_sentences(context, 2048)
                qa_pairs = [{'question': question, 'answer': answer} for question, answer in zip(questions, answers)]

            rows.extend(
                _qa_rows_from_entry_multi_agent(
                    chunks=chunks,
                    qa_pairs=qa_pairs,
                    base_sample_id=item_idx,
                    data_source=split_name,
                    sub_source=metadata.get("source"),
                    agent_ids=["answer_gen", "fact_split", "long_mem", "short_mem"],
                )
            )

        # TODO: missing chunk date synthesis process
        
    return _save_rows(rows, output_path)



# -------------------------
# 处理路径 2：基于 mem-alpha 产出的 chunk-level JSON/Parquet
# -------------------------

def process_memoryagentbench_from_memalpha(
    *,
    input_path: str,
    output_path: str = "./data/memoryagentbench_qa_from_memalpha.parquet",
    system_prompt: str = "You are a helpful memory-augmented assistant.",
    limit: int | None = None,
) -> str:

    pass # TODO: 实现 加载mem-alpha的数据集并拓展成QA-level数据集


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryAgentBench QA-level 预处理")
    parser.add_argument("--mode", choices=["raw", "memalpha"], required=True, help="处理路径")
    parser.add_argument("--split", default="Accurate_Retrieval", help="HF split（raw 模式）")
    parser.add_argument("--input_path", type=str, default="", help="memalpha 模式输入文件")
    parser.add_argument("--output_path", type=str, default="./data/memoryagentbench_qa.parquet", help="输出 parquet 路径")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，便于调试")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=2048, help="raw 模式 chunk 最大 token 数")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful memory-augmented assistant.", help="系统提示词")
    args = parser.parse_args()

    if args.mode == "raw":
        path = process_memoryagentbench_raw(
            split=args.split,
            output_path=args.output_path,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            limit=args.limit,
            system_prompt=args.system_prompt,
        )
    else:
        if not args.input_path:
            raise ValueError("memalpha 模式需要提供 --input_path")
        path = process_memoryagentbench_from_memalpha(
            input_path=args.input_path,
            output_path=args.output_path,
            system_prompt=args.system_prompt,
            limit=args.limit,
        )

    print(f"Saved QA-level dataset to: {path}")


if __name__ == "__main__":
    main()

