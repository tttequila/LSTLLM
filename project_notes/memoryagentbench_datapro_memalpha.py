# Standard library imports
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List, Any, Tuple

# Third-party imports
import dotenv
import nltk
import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset
from openai import AzureOpenAI
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

# Azure OpenAI configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
client = AzureOpenAI(
    api_key=api_key,
    api_version="2025-01-01-preview",
    azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
)

qwen32b_server_url = os.getenv("QWEN_URL")

# User message templates for conversational format
USER_MESSAGE_TEMPLATES = [
    "Here are some new facts I learnt just now:",
    "Here are some news I learnt earlier:",
    "The following are some more news I learnt:",
    "The following are some new knowledge:",
    "I just discovered some interesting information:",
    "Let me share some additional facts I found:",
    "Here's some more information I came across:",
    "I want to tell you about some new discoveries:",
    "There are some important details I learned:",
    "I have some fresh insights to share:",
    "I'd like to tell you about some recent findings:",
    "Let me update you with some new information:",
    "I've gathered some additional data to share:",
    "Here's some valuable information I collected:",
    "I want to share some knowledge I just acquired:",
    "Let me provide you with some new details:",
    "I have some interesting updates for you:",
    "Here are some facts I recently discovered:",
    "I'd like to share some important information:",
    "Let me tell you about some new developments:"
]

# User message templates for classification tasks
CLASSIFICATION_USER_TEMPLATES = [
    "Here are some classification examples to learn from. Please pay attention to the labels:",
    "I have some labeled classification examples for you to study:",
    "The following are classification examples with their corresponding labels:",
    "Please observe these classification examples and their associated labels:",
    "Here are training examples for classification. Note the labels carefully:",
    "I'm sharing some classification data with labels for you to learn:",
    "These are labeled examples for classification tasks:",
    "Please study these classification instances and their labels:",
    "Here are some examples with classification labels to remember:",
    "The following classification examples include important label information:",
    "I want you to learn from these classified examples:",
    "Here are categorized examples with their respective labels:",
    "Please memorize these classification examples and their labels:",
    "These labeled training examples are for classification:",
    "I'm providing classification data with labels for your reference:",
    "Study these classification examples and pay attention to the categories:",
    "Here are some annotated classification examples:",
    "Please learn from these labeled classification instances:",
    "The following are classification training examples with labels:",
    "These examples show different classes - please note the labels:"
]

# Assistant response templates
ASSISTANT_RESPONSE_TEMPLATES = [
    "Sure I will remember them.",
    "Got it. I will remember them.",
    "Thank you for sharing. I've noted this information.",
    "I understand. I'll keep this in mind.",
    "Thanks for the update. I've recorded these facts.",
    "Received. I'll store this information.",
    "Noted. I'll remember these details.",
    "I've processed this information and will remember it.",
    "Thanks for letting me know. I'll keep track of this.",
    "Perfect. I've stored this information in my memory.",
    "Understood. I'll keep these facts for future reference.",
    "Excellent. I've documented all of this information.",
    "Thanks for the information. I've saved it.",
    "Appreciated. I'll retain these important details.",
    "Great! I've added this to my knowledge base.",
    "I've successfully recorded all of this data.",
    "Wonderful. I'll remember these key points.",
    "Thanks for sharing. I've committed this to memory.",
    "I've captured all of this valuable information."
]

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The input format is:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

# ======
# 辅助函数
# ======

def save_processed_data(data, filename="processed_data.json"):
    """Save processed data to JSON file"""
    print(f"Saving {len(data)} instances to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {filename}")
    

def convert_json_to_parquet(json_filename, dataset_name):
    """Convert JSON file to parquet format and save in ./data/memalpha/"""

    # Create output directory
    output_dir = "./data/memalpha"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {json_filename} to parquet format...")

    # Load JSON data
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame-friendly format
    rows = []
    for i, instance in enumerate(data):
        # Handle both 'chunks' and 'context_chunks' keys
        chunks_key = 'chunks' if 'chunks' in instance else 'context_chunks'
        # Convert chunks list to string representation
        chunks_str = json.dumps(instance[chunks_key])

        if len(instance['questions_and_answers']) > 100:
            # random sample 100 questions_and_answers
            instance['questions_and_answers'] = random.sample(instance['questions_and_answers'], 100)

        # Convert questions_and_answers to string representation
        qa_str = json.dumps(instance['questions_and_answers'])

        row = {
            'instance_id': i,
            'prompt': instance.get('prompt', ''),  # Use get() with default for lme_train
            'chunks': chunks_str,
            'questions_and_answers': qa_str,
            'num_chunks': len(instance[chunks_key]),
            'num_questions': len(instance['questions_and_answers']),
        }

        if 'sub_source' in instance:
            row['sub_source'] = instance['sub_source']
        elif 'data_source' in instance:
            row['sub_source'] = instance['data_source']
        elif 'metadata' in instance:
            row['sub_source'] = instance['metadata']['source']

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save as parquet
    parquet_filename = os.path.join(output_dir, f"processed_{dataset_name}_data.parquet")
    df.to_parquet(parquet_filename, index=False)

    print(f"Parquet file saved to {parquet_filename}")
    print(f"Parquet file contains {len(df)} instances")

    return parquet_filename



def judge_answer_with_token_logic(ground_truth_answer, predicted_answer, debug=False):
    """
    判断答案是否正确，基于token计数和字符串包含逻辑

    Args:
        ground_truth_answer: 正确答案
        predicted_answer: 模型预测答案
        debug: 是否打印调试信息

    Returns:
        int: 0 if 应该保留该例子 (judge as incorrect), 1 if 应该移除该例子
    """
    # 将原始输入转换成str类型并去除前后空格
    ground_truth = str(ground_truth_answer).strip()
    predicted = str(predicted_answer).strip()


    if not ground_truth or not predicted:
        if debug:
            print(f"Empty answer detected - keeping example (GT: '{ground_truth}', Pred: '{predicted}')")
        return 0  # Keep examples with empty answers for safety

    # 计算正确答案的token数
    ground_truth_tokens = count_tokens(ground_truth)

    # 检查条件 (1): 正确答案的token数小于5
    condition_1 = ground_truth_tokens < 5

    # 检查条件 (2): 正确答案的小写形式不在预测答案的小写形式中
    condition_2 = ground_truth.lower() not in predicted.lower()

    # 如果两个条件都满足，则设置judge为0 (保留该例子)
    if condition_1 and condition_2:
        if debug:
            print(f"KEEP: Short answer ({ground_truth_tokens} tokens) not found in prediction")
            print(f"  GT: '{ground_truth}'")
            print(f"  Pred: '{predicted[:100]}...'")
        return 0  # Keep the example
    else:
        if debug:
            reason = []
            if not condition_1:
                reason.append(f"answer too long ({ground_truth_tokens} >= 5 tokens)")
            if not condition_2:
                reason.append("answer found in prediction")
            print(f"REMOVE: {', '.join(reason)}")
            print(f"  GT: '{ground_truth}'")
            print(f"  Pred: '{predicted[:100]}...'")
        return 1  # Remove the example


# VARIABLE CHUNK SIZE FEATURE:
# 对于SQuAD, HotpotQA, WOS46985, PubMed-RCT, ArXiv-Classification, and EurLex数据集,
# chunks现在有100到4096个token之间的可变大小，而不是固定大小约为2000个token。
# 这由chunking函数中的variable_size参数控制。
# Booksum数据集不包括在这个功能中。

def count_tokens(text, model="gpt-4o-mini"):
    """使用tiktoken计算token数
    
    Args:
        text: 文本
        model: 模型

    Returns:
        int: token数
    """
    encoding = tiktoken.encoding_for_model(model)

    # 将输入转换为字符串，如果输入不是字符串
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        else:
            text = str(text)

    return len(encoding.encode(text))

def create_chunks_use_sent_tokenizer(text, max_tokens=10000):
    """对于当前输入的文本，创建chunks，使用句子分词器
    Args:
        text: 文本
        max_tokens: 最大token数

    Returns:
        List[str]: chunks
    """
    # Make sure we have the punkt tokenizer downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # 使用nltk的句子分词器将文本分割成句子
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    # 遍历句子
    for sentence in sentences:
        # 如果句子中包含<|endoftext|>，则替换为\n
        if '<|endoftext|>' in sentence:
            sentence = sentence.replace('<|endoftext|>', '\n')

        # 计算句子token数
        sentence_tokens = count_tokens(sentence)

        # 在保证句子语义完整性的前提下，将整个文本分割成以句子为单位的chunks
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            if current_chunk:
                # 在句子之间添加空格
                current_chunk += " " + sentence
                current_tokens += sentence_tokens + count_tokens(" ")
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens

    # 添加最后一个chunk进行收尾
    if current_chunk:
        chunks.append(current_chunk.strip())

    # 返回分割好的chunks
    return chunks

def create_chunks(contexts, max_tokens=2000, min_tokens=None, variable_size=False):
    """对于当前输入的contexts，创建chunks，确保每个chunk的token数小于max_tokens

    Args:
        contexts: 上下文列表
        max_tokens: 每个chunk的最大token数 (默认: 2000)
        min_tokens: 每个chunk的最小token数 (当variable_size=True时使用，默认: max_tokens/20)
        variable_size: 如果为True，则随机改变每个chunk的大小，范围在min_tokens和max_tokens之间
    """
    chunks = []
    current_chunk = ""
    current_tokens = 0

    # 设置chunk的大小区间
    # 如果min_tokens未提供且variable_size为True，则设置默认min_tokens为max_tokens/20
    if variable_size and min_tokens is None:
        min_tokens = max(100, max_tokens // 20)
    # 如果variable_size为True，则设置target_tokens为随机值，范围在min_tokens和max_tokens之间
    if variable_size:
        target_tokens = random.randint(min_tokens, max_tokens) # 随机值
    else:
        target_tokens = max_tokens # 固定值

    # 遍历contexts
    for context in contexts:
        context_tokens = count_tokens(context) # 计算context的token数

        # 如果添加这个context会超过target_tokens，则开始一个新的chunk
        if current_tokens + context_tokens > target_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = context
            current_tokens = context_tokens

            # Set new target for next chunk if using variable size
            if variable_size:
                target_tokens = random.randint(min_tokens, max_tokens) # 重新采样随机值
            else:
                target_tokens = max_tokens # 固定值
        else:
            if current_chunk:
                current_chunk += "\n\n" + context # 在句子之间添加空行
                current_tokens += context_tokens + count_tokens("\n\n")
            else:
                current_chunk = context # 设置当前chunk为当前context
                current_tokens = context_tokens

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def batch_process_questions_with_qwen32b(questions, batch_size=32, system_prompt=None, model="qwen3-32b", no_thinking=False):
    """
    使用Qwen32B模型批量处理问题

    Args:
        questions: 问题列表
        batch_size: 每个batch的问题数量
        model: 使用的模型

    Returns:
        List of responses corresponding to each question
    """

    # Import and setup Qwen client
    from openai import OpenAI
    from transformers import AutoTokenizer
    import time

    # Setup Qwen client
    client = OpenAI(
        base_url=qwen32b_server_url,
        api_key="EMPTY"
    )

    # 初始化tokenizer用于prompt转换
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    print(f"Starting batch processing of {len(questions)} questions with Qwen32B, batch size {batch_size}")

    all_responses = []

    # Process questions in batches
    for i in range(0, len(questions), batch_size):
        # 获取当前batch
        batch_questions = questions[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(questions) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")

        # 批量转换questions为prompts
        batch_prompts = []
        for question in batch_questions:
            # 根据每个问题，让模型生成回答
            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            else:
                messages = [
                    {"role": "user", "content": question}
                ]

            # Convert to prompt using tokenizer
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            if no_thinking:
                prompt += "<think></think>\n\n"

            batch_prompts.append(prompt)

        # 使用completions API处理整个batch
        response = client.completions.create(
            model=model,
            prompt=batch_prompts,
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        # 提取响应
        batch_responses = [choice.text for choice in response.choices]

        if not no_thinking:
            # 需要移除<think></think>标签
            batch_responses = [(x.split("</think>")[1] if "</think>" in x else x) for x in batch_responses]

        # 将响应添加到all_responses中
        all_responses.extend(batch_responses)
        print(f"Completed batch {batch_num}/{total_batches}")
        # Delay between batches to avoid overloading the server
        if i + batch_size < len(questions):
            time.sleep(0.5)

    print(f"Batch processing complete. Generated {len(all_responses)} responses.")
    # 返回所有响应
    return all_responses




# ===============================================
# 处理MemoryAgentBench数据集（Accurate Retrieval）
# 原始字段：context, questions, answers，metadata
#      context: 一个很长的字符串，多文档拼接而成
#      questions: 一个List[str]，包含多个问题
#      answers: 一个List[List[str]]，每个问题提供了复数个可选答案
#      其中AR和TTL的metadata包括：demo，haystack_sessions，keypoints，previous_events，qa_pair_ids, question_dates, question_ids, question_types, source
#      而LRU的metadata包括：haystack_sessions，keypoints，previous_events，qa_pair_ids, question_dates, question_ids, question_types, source
# 处理后的字段：prompt, chunks, questions_and_answers, data_source
# ===============================================
def process_memory_agent_bench(split='Accurate_Retrieval'):

    ar_ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)

    # Load the Qwen tokenizer for accurate token counting for longmemeval_s*
    from transformers import AutoTokenizer
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    def count_tokens_qwen(text):
        """Count tokens using Qwen3-32B tokenizer"""
        return len(qwen_tokenizer.encode(text))

    processed_data = []

    if split == 'Long_Range_Understanding':
        with open("./data/memoryagentbench_answers_to_keywords_mapping.json", 'r') as file:
            answers_to_keywords_mapping = json.load(file)

    # 遍历MemoryAgentBench的每个实例，处理每个实例的context，questions，answers
    for item_idx, item in tqdm(enumerate(ar_ds), desc=f"Processing {split}", total=len(ar_ds)):
        # 过滤掉部分数据来源
        if 'niah' in item['metadata'].get('source', ''):
            continue
        if item['metadata']['source'] in ['longmemeval_s_-1_500', 'eventqa_65536', 'eventqa_131072', 'eventqa_full', 'infbench_qa_eng_shots2']:
            continue
        if item['metadata']['source'] in ['recsys_redial_full']:
            continue

        # if item['metadata']['source'] in ['icl_trec_coarse_6600shot_balance', 'icl_nlu_8296shot_balance']:
        #     continue

        context = item['context']
        questions = item['questions']
        answers = item['answers']

        # Create chunks from the context using sentence tokenization

        # ===============================================
        # 处理数据来源为longmemeval_s*的实例（该数据来源的context是交替的 [timestamp, session, timestamp, session, ...] 列表）
        # ===============================================
        # 处理数据来源为longmemeval_s*的实例（该数据来源的context是交替的 [timestamp, session, timestamp, session, ...] 列表）
        if item['metadata']['source'] == 'longmemeval_s*':
            # Context is an alternating list: [timestamp:str, session:list, timestamp:str, session:list, ...]
            # Each session is a list of turns with keys 'role' and 'content'.
            # We will render each pair into text and then group into chunks with at least 2048 tokens.
            try:
                all_context = eval(context) if isinstance(context, str) else context
            except Exception:
                all_context = json.loads(context)

            assert isinstance(all_context, list), "Expected all_context to be a list"
            assert len(all_context) % 2 == 0, "Expected alternating [timestamp, session] pairs"
            
            # 遍历all_context，将每个[timestamp, session]转换成文本
            for idx in range(0, len(all_context), 2):
                ts = all_context[idx]
                session = all_context[idx + 1]
                assert isinstance(ts, str), f"Expected timestamp at index {idx} to be str"
                assert isinstance(session, list), f"Expected session at index {idx+1} to be list"
                for turn in session:
                    # 确保每轮对话都是chat模板
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
                if count_tokens_qwen(f"<{role.capitalize()}>{content}") <= max_tokens - 150:  # Leave some buffer for headers
                    return [test_turn]

                # 否则将对话拆分成多个句子，然后根据句子长度拆分成多个chunks
                import re
                # 先尝试按句子拆分
                sentences = re.split(r'(?<=[.!?])\s+', content)

                split_turns = []
                current_content = ""

                # 遍历每个句子，将句子拆分成多个chunks
                for sentence in sentences:
                    test_content = current_content + (" " if current_content else "") + sentence
                    test_text = f"<{role.capitalize()}>{test_content}"

                    # 如果句子长度足够小，则直接加入当前chunk
                    if count_tokens_qwen(test_text) <= max_tokens - 150:
                        current_content = test_content
                    # 如果句子长度超过最大长度，则将当前chunk保存，并开始新的chunk
                    else:
                        if current_content:
                            # Save current chunk
                            split_turns.append({'role': role, 'content': current_content.strip()})
                            current_content = sentence
                        else:
                            # Single sentence is too long, need to split it further
                            words = sentence.split()
                            chunk = ""
                            for word in words:
                                test_chunk = chunk + (" " if chunk else "") + word
                                if count_tokens_qwen(f"<{role.capitalize()}>{test_chunk}") <= max_tokens - 150:
                                    chunk = test_chunk
                                else:
                                    if chunk:
                                        split_turns.append({'role': role, 'content': chunk.strip()})
                                    chunk = word
                            if chunk:
                                current_content = chunk

                # 如果还有剩余内容，则加入当前chunk
                if current_content:
                    split_turns.append({'role': role, 'content': current_content.strip()})

                return split_turns

            def split_session_into_segments(timestamp_str, turns):
                """将对话拆分成多个segments，每个segment包含一个角色和一段对话"""
                # Each segment should be <= 2048 tokens
                segments = []
                current_turns = []

                # 遍历每个对话轮次，将每个对话轮次拆分成多个chunks
                for turn in turns:
                    # 检查单个对话轮次是否超过最大长度
                    single_turn_text = f"<{turn['role'].capitalize()}>{turn['content']}"
                    # 如果单个对话轮次超过最大长度，则将对话轮次拆分成多个chunks
                    if count_tokens_qwen(single_turn_text) > 2048 - 150:  # Leave buffer for header
                        # 将对话轮次拆分成多个chunks，返回轮次列表
                        split_turns = split_large_turn(turn, max_tokens=2048)

                        # 遍历每个拆分后的对话轮次，将每个对话轮次添加到当前segments
                        for split_turn in split_turns:
                            # 尝试将每个拆分后的对话轮次添加到当前segments
                            test_turns = current_turns + [split_turn]
                            is_cont = len(segments) > 0
                            # 渲染对话轮次，返回文本
                            candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)

                            # 如果渲染后的文本长度足够小，则直接加入当前segments
                            if count_tokens_qwen(candidate_text) <= 2048:
                                current_turns = test_turns
                            # 如果渲染后的文本长度超过最大长度，则将当前segments保存，并开始新的segments
                            else:
                                # Flush current segment and start new one
                                if current_turns:
                                    # Check if last turn is User - if so, move it to next segment
                                    if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                        last_turn = current_turns.pop()
                                        segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                        current_turns = [last_turn, split_turn]
                                    else:
                                        segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                        current_turns = [split_turn]
                                else:
                                    current_turns = [split_turn]
                    # 如果单个对话轮次不超过最大长度，则直接加入当前segments
                    else:
                        # 将当前对话轮次添加到当前segments
                        test_turns = current_turns + [turn]
                        is_cont = len(segments) > 0
                        candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)

                        # 如果渲染后的文本长度足够小，则直接加入当前segments
                        if count_tokens_qwen(candidate_text) <= 2048:
                            # Turn fits, add it to current segment
                            current_turns = test_turns
                        # 如果渲染后的文本长度超过最大长度，则将当前segments保存，并开始新的segments
                        else:
                            # Adding this turn would exceed limit
                            if current_turns:
                                # Check if last turn is User - if so, move it to next segment
                                if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                    last_turn = current_turns.pop()
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [last_turn, turn]
                                else:
                                    # Flush current segment and start new one with this turn
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [turn]
                            else:
                                # This shouldn't happen now since we split large turns
                                segments.append(render_session(timestamp_str, [turn], continuation=is_cont))
                                current_turns = []

                # 如果还有剩余对话轮次，则加入当前segments
                if current_turns:
                    # Check if last turn is User - if so, we might want to keep it with previous content
                    # For the final segment, we'll allow User turns since there's no next segment
                    segments.append(
                        render_session(timestamp_str, current_turns, continuation=len(segments) > 0)
                    )

                return segments

            # 使用前面定义得函数，将每个session拆分成多个segments，然后聚合成多个chunks
            # 目标是每个chunk大约2048个token，但最大不超过2048个token
            chunks = []
            current_chunk = ""
            # 遍历所有原始对话轮次
            for idx in range(0, len(all_context), 2):
                ts = all_context[idx]
                session_turns = all_context[idx + 1]
                # 调用split_session_into_segments函数，将每个session拆分成多个segments
                session_segments = split_session_into_segments(ts, session_turns)
                # 遍历当前session的每个segment，组织成chunk
                for segment_text in session_segments:
                    # 如果当前chunk有内容，则续上
                    if current_chunk:
                        candidate = current_chunk + "\n\n" + segment_text
                    # 如果当前chunk没有内容，则直接赋值
                    else:
                        candidate = segment_text
                    # 计算当前chunk的token数量
                    candidate_tokens = count_tokens_qwen(candidate)

                    # 拆分过大candidate
                    if candidate_tokens > 2048:
                        # 如果当前chunk的token数量超过最大长度，则将当前chunk保存，并开始新的chunk
                        if current_chunk:
                            chunks.append(current_chunk)
                        # Start new chunk with this segment
                        current_chunk = segment_text
                    # 如果当前candidate的token数量接近目标长度，则直接加入当前chunk,重开一个新的chunk
                    elif candidate_tokens >= 1800:  # Close enough to target, make it a chunk
                        chunks.append(candidate)
                        current_chunk = ""
                    # 如果还有剩余空间，直接将candidare赋值为当前chunk
                    else:
                        # Still room to add more
                        current_chunk = candidate

            # 上一步中未重开新的chunk
            if current_chunk:
                # Check if the last chunk ends with a User turn
                # We'll inspect the chunk to see if it ends with "<User>"
                lines = current_chunk.strip().split('\n')
                if lines and lines[-1].startswith('<User>'):
                    # If possible, we should have moved this to the next chunk
                    # But since this is the last chunk, we'll keep it
                    pass
                chunks.append(current_chunk)

            # 确保每个chunk的token数量不超过2048
            for chunk in chunks:
                assert count_tokens(chunk) <= 2048

        # ===============================================
        # 预分块TTL类数据
        # ===============================================
        elif split == 'Test_Time_Learning':
            # 直接调用create_chunks_use_sent_tokenizer函数，将整块的context拆分成多个较小的chunks
            chunks = create_chunks_use_sent_tokenizer(context, max_tokens=1024)

        else:
            chunks = create_chunks_use_sent_tokenizer(context, max_tokens=2048)

        # ===============================================
        # 处理question和answer字段
        # ===============================================
        questions_and_answers = []
        for q, a in zip(questions, answers):
            questions_and_answers.append({'question': q, 'answer': a})


        # ===============================================
        # 根据数据划分整体格式化数据实例
        # ===============================================
        if split == 'Long_Range_Understanding':
            # 使用BookSum-style date-based template 处理Long_Range_Understanding数据
            base_date = datetime(2024, 1, 1)
            formatted_chunks = []
            chunk_dates = []
            current_date = base_date

            # 将每个chunk包装成递增日期的对话轮次:
            # 
            # [Event happened on YYYY‑MM‑DD The user is reading a book]
            # <User> chunk_content
            # <System> Please remember what the user reads on YYYY‑MM‑DD …

            for chunk_idx, chunk_content in enumerate(chunks):
                # Create progressive dates (incrementing by random 1-3 days)
                if chunk_idx > 0:  # Don't add days for the first chunk
                    days_to_add = random.randint(1, 3)
                    current_date = current_date + timedelta(days=days_to_add)
                date_str = current_date.strftime("%Y-%m-%d")
                chunk_dates.append(date_str)

                # Format with the BookSum template
                formatted_chunk = f"[Event happened on {date_str} The user is reading a book]\n<User> {chunk_content}\n\n<System> Please remember what the user reads on {date_str}, save the details within the book, and retain a summary of the book the user has read so far."
                formatted_chunks.append(formatted_chunk)

            # 给questions and answers更新时间戳, prompt变为让模型总结给定时间窗口的书籍内容
            if chunk_dates:
                first_date = chunk_dates[0]
                last_date = chunk_dates[-1]
                updated_questions_and_answers = []
                for qa in questions_and_answers:
                    updated_qa = qa.copy()
                    # updated_qa['question'] = f"Based on the content I read from {first_date} to {last_date}, {qa['question']}"
                    updated_qa['question'] = f"Summarize the content of the book I read from {first_date} to {last_date}."
                    # 使用answers_to_keywords_mapping中的关键词作为answer
                    updated_qa['answer'] = answers_to_keywords_mapping[str(item_idx)]['keywords']
                    updated_questions_and_answers.append(updated_qa)
                questions_and_answers = updated_questions_and_answers

            chunks = formatted_chunks

        elif split == 'Accurate_Retrieval' and item['metadata']['source'] == 'infbench_qa_eng_shots2':
            # 使用BookSum-style date-based template 处理infbench_qa_eng_shots2数据
            # 在问题前加上 “Based on the content I read from start_date to end_date” 作为提示
            base_date = datetime(2024, 1, 1)
            formatted_chunks = []
            chunk_dates = []
            current_date = base_date

            for chunk_idx, chunk_content in enumerate(chunks):
                # Create progressive dates (incrementing by random 1-3 days)
                if chunk_idx > 0:  # Don't add days for the first chunk
                    days_to_add = random.randint(1, 3)
                    current_date = current_date + timedelta(days=days_to_add)
                date_str = current_date.strftime("%Y-%m-%d")
                chunk_dates.append(date_str)

                # Format with the BookSum template
                formatted_chunk = f"[Event happened on {date_str} The user is reading a book]\n<User> {chunk_content}\n\n<System> Please remember what the user reads on {date_str}, save the details within the book, and retain a summary of the book the user has read so far."
                formatted_chunks.append(formatted_chunk)

            # Update questions to use date ranges for infbench_qa_eng_shots2
            if chunk_dates:
                first_date = chunk_dates[0]
                last_date = chunk_dates[-1]
                updated_questions_and_answers = []
                for qa in questions_and_answers:
                    updated_qa = qa.copy()
                    updated_qa['question'] = f"Based on the content I read from {first_date} to {last_date}, {qa['question']}"
                    updated_questions_and_answers.append(updated_qa)
                questions_and_answers = updated_questions_and_answers

            chunks = formatted_chunks

        elif split == 'Accurate_Retrieval' and item['metadata']['source'] != 'longmemeval_s*' and item['metadata']['source'] != 'infbench_qa_eng_shots2':
            # 使用对话模板处理其他来源的Accurate_Retrieval数据
            # [Dialogue between User and Assistant on YYYY‑MM‑DD HH:MM]
            # <User>user_template
            # chunk_content
            # <Assistant>assistant_template

            base_date = datetime(2024, 1, 1)
            formatted_chunks = []

            for chunk_idx, chunk_content in enumerate(chunks):
                # Format using conversational template with random selection
                user_template = random.choice(USER_MESSAGE_TEMPLATES)
                assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

                # Create a timestamp for each chunk (incrementing by days)
                chunk_date = base_date + timedelta(days=chunk_idx)
                timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

                # Format the chunk with conversational template
                formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n{chunk_content}\n<Assistant>{assistant_template}"
                formatted_chunks.append(formatted_chunk)

            chunks = formatted_chunks

        elif split == 'Test_Time_Learning':
            # 使用分类的user模板,这部分子集数据主要执行fwe shot classification任务
            base_date = datetime(2024, 1, 1)
            formatted_chunks = []

            # 再前一步,对于TTL子集已经将context根据句子拆分成了多个chunks,这里需要将每个chunk再根据句子拆分
            for chunk_idx, chunk_content in enumerate(chunks):
                # Format using classification-specific templates
                user_template = random.choice(CLASSIFICATION_USER_TEMPLATES)
                assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

                # Create a timestamp for each chunk (incrementing by days)
                chunk_date = base_date + timedelta(days=chunk_idx)
                timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")


                new_chunk_content = ''
                # 遍历chunk内的段落(\n\n分割)
                for x in chunk_content.split("\n\n"):
                    try:
                        # 尝试取出label部分
                        sentence, label = x.split("label:")
                    except:
                        # 如果无法取出label部分，则说明该段落完全是上下文内容
                        continue
                    sentence = sentence.strip()
                    if len(sentence) == 0:
                        continue
                    label = label.strip()
                    # 重新将label拼在末尾
                    new_chunk_content += f"Sentence: {sentence}\nLabel: {label}\n\n"
                chunk_content = new_chunk_content.strip()

                # Format the chunk with classification template
                formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n{chunk_content}\n<Assistant>{assistant_template}"
                formatted_chunks.append(formatted_chunk)

            chunks = formatted_chunks

        # Create the data instance
        data_instance = {
            'prompt': 'I will provide you with sequential information chunks. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
            'chunks': chunks,
            'questions_and_answers': questions_and_answers,
            'data_source': 'accurate_retrieval' if split != 'Long_Range_Understanding' else 'long_range_understanding',
            'sub_source': item['metadata']['source']
        }

        # Add reading dates for long_range_understanding
        if split == 'Long_Range_Understanding' and chunk_dates:
            data_instance['reading_dates'] = {'start': chunk_dates[0], 'end': chunk_dates[-1]}

        processed_data.append(data_instance)

    return processed_data


def main():
    parser = argparse.ArgumentParser(description='Process datasets for memory training')
    parser.add_argument('--dataset', type=str, choices=['squad', 'squad_test', 'hotpotqa', 'booksum', 'friends', 'wos46985', 'pubmed-rct', 'arxiv-classification', 'eurlex', 'accurate_retrieval', 'long_range_understanding', 'conflict_resolution', 'test_time_learning', 'detectiveqa', 'lamp4', 'perltqa', 'narrativeqa', 'ttl_train', 'cr_train', 'lme_train'],
                       default='squad', help='Dataset to process (default: squad)')
    parser.add_argument('--convert-to-parquet', action='store_true',
                       help='Convert existing JSON files to parquet format')
    parser.add_argument('--split-train-test', action='store_true',
                       help='Combine all parquet files, shuffle, and split into train/test sets (80/20)')
    parser.add_argument('--split-single-dataset', action='store_true',
                       help='Split a single dataset into train/test sets (works with --dataset)')
    parser.add_argument('--filter-dataset', action='store_true',
                       help='Filter out questions that GPT-4o-mini can answer without context (squad/hotpotqa only)')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Maximum number of questions to test for filtering (for debugging)')
    parser.add_argument('--num-processes', type=int, default=16,
                       help='Number of parallel processes for filtering (default: 16)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio of data to use for training when splitting (default: 0.9)')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild the dataset even if it already exists')
    parser.add_argument('--split-by-difficulty', action='store_true',
                       help='Split training dataset by difficulty based on accuracy bins from analyze_results.py')
    parser.add_argument('--merge-datasets', type=str, nargs='+',
                       help='Create memalpha train/test files from specified datasets. Accepts multiple dataset names (e.g., booksum pubmed-rct)')
    parser.add_argument('--status', type=str, default='all',
                       help='Status of the dataset to process (default: all)')
    parser.add_argument('--output-name', type=str, default='memalpha',
                       help='Name of the output dataset (default: memalpha)')
    parser.add_argument('--limit-size', type=int, default=None,
                       help='Maximum number of examples per training dataset (default: None)')

    args = parser.parse_args()


    # If splitting by difficulty
    if args.split_by_difficulty:
        print("Splitting training dataset by difficulty based on accuracy bins...")
        output_files = split_dataset_by_difficulty()
        if output_files:
            print(f"Successfully split dataset into {len(output_files)} difficulty levels!")
        else:
            print("Failed to split dataset by difficulty.")
        return

    # If creating memalpha from specified datasets
    if args.merge_datasets:
        print(f"Creating memalpha from datasets {args.merge_datasets}...")
        train_path, test_path = merge_into_memalpha(args.merge_datasets, random_seed=42, status=args.status, output_name=args.output_name, limit_size=args.limit_size)
        if train_path and test_path:
            print(f"\nSuccessfully created {args.output_name} from specified datasets!")
        else:
            print("Failed to create memalpha from specified datasets.")
        return

    # If filtering dataset
    if args.filter_dataset:
        if args.dataset not in ['squad', 'hotpotqa']:
            print(f"Error: Filtering only supported for 'squad' and 'hotpotqa', got '{args.dataset}'")
            return

        print(f"Filtering {args.dataset} dataset with {args.num_processes} processes...")
        output_file = filter_dataset(args.dataset, max_questions_to_test=args.max_questions, num_processes=args.num_processes)
        if output_file:
            print(f"Successfully filtered {args.dataset} dataset!")
        else:
            print(f"Failed to filter {args.dataset} dataset.")
        return

    # If combining and splitting datasets
    if args.split_train_test:
        print("Combining and splitting datasets into train/test sets...")
        train_path, test_path = combine_and_split_datasets(train_ratio=args.train_ratio)
        if train_path and test_path:
            print(f"\nSuccessfully created train/test split!")
        else:
            print("Failed to create train/test split.")
        return

    # If splitting a single dataset
    if args.split_single_dataset:
        print(f"Splitting {args.dataset} dataset into train/test sets...")
        if args.dataset == 'perltqa':
            train_path, test_path = split_dataset(args.dataset, train_ratio=0.9)
        elif args.dataset == 'lme_train':
            train_path, test_path = split_dataset(args.dataset, train_ratio=1.0)
        else:
            train_path, test_path = split_dataset(args.dataset, train_ratio=args.train_ratio)
        if train_path and test_path:
            print(f"\nSuccessfully created train/test split for {args.dataset}!")
        else:
            print(f"Failed to create train/test split for {args.dataset}.")
        return

    # Process the dataset based on the argument
    elif args.dataset == 'accurate_retrieval':
        filename = './data/processed_accurate_retrieval_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_memory_agent_bench("Accurate_Retrieval")
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'long_range_understanding':
        filename = './data/processed_long_range_understanding_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_memory_agent_bench("Long_Range_Understanding")
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'test_time_learning':
        filename = './data/processed_test_time_learning_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_memory_agent_bench("Test_Time_Learning")
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    else:
        print(f"Unknown dataset: {args.dataset}")
        return

    # Save to file (only if we have data)
    if processed_data:

        # Print statistics
        print_statistics(processed_data, args.dataset)

        # Automatically convert to parquet
        print("\nConverting to parquet format...")
        try:
            parquet_file = convert_json_to_parquet(filename, args.dataset)
            print(f"Successfully created parquet file: {parquet_file}")
        except Exception as e:
            print(f"Error converting to parquet: {e}")
    else:
        print(f"No data processed for {args.dataset} dataset.")

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Start method already set
    main()
