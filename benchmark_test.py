from pathlib import Path

from datasets import load_dataset  # type: ignore[import-error]

memory_agent_bench_path = Path(r"C:/Users/TTTeq/Documents/CodeFolder/CP/VCS/Dataset/MemoryAgentBench")
split_name = "Accurate_Retrieval"
remote_dataset_id = "ai-hyz/MemoryAgentBench"
download_root = "C:/Users/TTTeq/Documents/CodeFolder/CP/VCS/Dataset/MemoryAgentBench"  # 填写下载目录后再运行脚本


def ensure_local_dataset(
    local_path: Path, dataset_id: str, split: str, download_cache_dir: str
) -> Path:
    """确保指定的本地数据集路径存在, 否则从 Hugging Face 下载并保存。

    Args:
        local_path: 最终希望存放数据集的本地路径。
        dataset_id: Hugging Face 上的数据集标识符。
        split: 需要加载的 split 名称, 同时用于下载时的 split。
        download_cache_dir: 下载缓存目录, 为空时禁止自动下载，防止误操作。

    Returns:
        Path: 一定存在的数据集路径。
    """
    if local_path.exists():
        return local_path
    if not download_cache_dir:
        raise RuntimeError(
            "数据集路径不存在，请先填写 download_root 或手动下载数据集到 %s" % local_path
        )
    cache_path = Path(download_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_id, cache_dir=str(cache_path))
    local_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(local_path))
    return local_path


print("spot1")

longme_eval_path = ensure_local_dataset(
    memory_agent_bench_path, remote_dataset_id, split_name, download_root
)
longme_eval_split = load_dataset(str(longme_eval_path), split=split_name)

print('spot2')

def find_longmemeval_sample(dataset):
    """Return the first streaming sample whose metadata.source mentions "longmemeval".

    Args:
        dataset: A streaming split that yields samples with a metadata dict.

    Returns:
        dict: The matching example.
    """
    for i, example in enumerate(dataset):
        print(f'checking {i}-th sample')
        metadata = example.get("metadata", {})
        source = metadata.get("source", "")
        if "longmemeval" in source.lower():
            return example
    raise LookupError("No sample with metadata.source containing 'longmemeval' found in split %s" % split_name)

longmemeval_sample = find_longmemeval_sample(longme_eval_split)
print('spot3')
print(len(longmemeval_sample['context']))