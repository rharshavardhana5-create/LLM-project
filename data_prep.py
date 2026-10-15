"""
data_prep.py

Loads the `ronantakizawa/github-codereview` dataset (167K+ real
(before_code, reviewer_comment, after_code) triplets from top GitHub repos,
plus ~51K "clean code, no comment needed" negatives), filters it down to a
target language + quality threshold, and writes train/val/test JSONL files
in chat-instruction format, ready for SFT.

This is the standalone / reproducible version of the data step. The Colab
notebook (train_qlora.ipynb) inlines an equivalent version of this logic so
it's self-contained, but you can also run this locally or on a CI box to
version your exact dataset snapshot.

Usage:
    pip install datasets
    python data_prep.py --language Python --max_examples 6000

Output:
    data/train.jsonl
    data/val.jsonl
    data/test.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "ronantakizawa/github-codereview"

SYSTEM_PROMPT = (
    "You are an expert code reviewer. You will be shown a chunk of code "
    "from a pull request, with its language and file path. Write the "
    "single most important review comment a senior engineer would leave. "
    "Be specific and actionable. If the code has no issues worth "
    "flagging, respond with exactly: No issues found."
)


def build_example(row: dict) -> dict:
    """Turn one dataset row into a chat-format SFT example."""
    user_msg = (
        f"File: {row['file_path']}\n"
        f"Language: {row['language']}\n\n"
        f"Diff:\n{row['diff_context']}\n\n"
        f"Code chunk:\n{row['before_code']}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": row["reviewer_comment"].strip()},
        ],
        # kept for evaluate.py — negative-detection metrics, error analysis by repo/language
        "meta": {
            "repo_name": row["repo_name"],
            "language": row["language"],
            "comment_type": row.get("comment_type"),
            "is_negative": row["is_negative"],
            "quality_score": row["quality_score"],
        },
    }


def select_split(rows, cap: int, negative_frac: float, min_quality: float, rng: random.Random):
    positives = [r for r in rows if not r["is_negative"] and r["quality_score"] >= min_quality]
    negatives = [r for r in rows if r["is_negative"]]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_neg = int(cap * negative_frac)
    n_pos = cap - n_neg
    chosen = positives[:n_pos] + negatives[:n_neg]
    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="Python")
    parser.add_argument("--min_quality", type=float, default=0.3)
    parser.add_argument("--max_examples", type=int, default=6000,
                         help="Cap on the training set so a run fits a free Colab T4 session.")
    parser.add_argument("--negative_frac", type=float, default=0.15,
                         help="Fraction of each split that should be 'No issues found' negatives.")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset(DATASET_NAME)

    caps = {"train": args.max_examples, "validation": 400, "test": 500}
    out_names = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, cap in caps.items():
        rows = ds[split].filter(lambda r: r["language"] == args.language)
        chosen = select_split(rows, cap, args.negative_frac, args.min_quality, rng)
        examples = [build_example(r) for r in chosen]

        with open(out_dir / out_names[split], "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        n_neg = sum(1 for r in chosen if r["is_negative"])
        print(f"{split}: {len(examples)} examples ({n_neg} negative) -> {out_dir / out_names[split]}")


if __name__ == "__main__":
    main()
