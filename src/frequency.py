import argparse
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import IterableDataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging


def tokenfreq(dataset, tokenizer, worker_id=0):
    freq_counter = Counter([])

    for instance in tqdm(dataset, position=worker_id, desc=f"{worker_id}"):
        text = instance["text"]
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"]
        freq_counter += Counter(input_ids)
    return freq_counter


def bitokenfreq(dataset, tokenizer, worker_id=0):
    freq_counter = Counter([])

    for instance in tqdm(
        dataset, position=worker_id, desc=f"{worker_id}", mininterval=2.0
    ):
        text = instance["text"]
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )["input_ids"]

        freq_counter.update(Counter(zip(input_ids[:-1], input_ids[1:])))
    return freq_counter


def wordfreq(dataset, tokenizer, worker_id=0):
    freq_counter = Counter([])

    for instance in tqdm(
        dataset,
        position=worker_id,
        desc=f"{worker_id}",
    ):
        text = instance["text"]
        words = re.split("[, +/-]", text)
        freq_counter += Counter(words)
    return freq_counter


def get_dataset(dataset_name: str) -> IterableDataset:
    if dataset_name == "wikitext103":
        return load_dataset(
            "salesforce/wikitext",
            name="wikitext-103-raw-v1",
            split="test",
            streaming=True,
        )
    if dataset_name == "openwebtext":
        return load_dataset(
            "openwebtext", trust_remote_code=True, split="train", streaming=True
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main(dataset_name: str, mode: str, tokenizer_name: str = None, num_workers=1):
    dataset = get_dataset(dataset_name)
    print("dataset: ", dataset)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    save_path = Path(f"freqs/{mode}/{dataset_name}_{tokenizer_name}.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    freq_func = {"word": wordfreq, "token": tokenfreq, "bitoken": bitokenfreq}[mode]

    if num_workers > 1:
        import multiprocessing

        with multiprocessing.Pool(num_workers) as pool:
            freq_counters = pool.starmap(
                freq_func,
                [
                    (
                        dataset.shard(num_shards=dataset.num_shards, index=shard),
                        tokenizer,
                        shard,
                    )
                    for shard in range(dataset.num_shards)
                ],
            )
    else:
        freq_counters = [freq_func(dataset, tokenizer)]

    freq_counter = Counter([])
    for counter_id in tqdm(range(len(freq_counters)), desc="Aggregating..."):
        freq_counter.update(freq_counters[counter_id])
        freq_counters[counter_id] = Counter([])

    print("Saving to ", save_path)
    torch.save(freq_counter, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data", default="wiki")  # wiki, books, or openwebtext
    parser.add_argument("--tokenizer", default=None)  # bert-base-cased or gpt2
    parser.add_argument("--mode", choices=["word", "token", "bitoken"], default="word")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    # not display warnings
    logging.set_verbosity(40)

    main(args.data, args.mode, args.tokenizer, args.num_workers)
