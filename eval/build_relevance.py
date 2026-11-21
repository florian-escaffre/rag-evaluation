"""Build proxy relevance labels: how many corpus passages contain each ground truth.

Needed for Recall@K (denominator). Run once, the result is reused by run_eval.py:

    python eval/build_relevance.py

Assumption to verify: the index is built from the dataset passages as they are
(no chunking). If your index splits passages into chunks, scan those chunks
instead, otherwise Recall@K can exceed 1.
"""

import json
import sys
from pathlib import Path

# Make the project root importable (for `config`, `eval`)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm

import config
from eval.metrics import normalize

RELEVANCE_PATH = config.ROOT / "eval" / "relevance.json"


def main() -> None:
    questions = list(load_dataset(config.DATASET_NAME, "question-answer")["test"])
    # Config and split names: check them on the dataset page if loading fails
    passages = list(load_dataset(config.DATASET_NAME, "text-corpus")["passages"])

    # Normalize and pad every passage once: padding with spaces gives
    # whole-word matching, same rule as metrics.contains
    padded = [f" {normalize(p['passage'])} " for p in passages]

    relevance = {}
    for item in tqdm(questions):
        truth = normalize(item["answer"])
        # Yes/no questions have no passage to look for
        if truth in {"yes", "no", ""}:
            continue
        needle = f" {truth} "
        relevance[item["question"]] = sum(needle in text for text in padded)

    RELEVANCE_PATH.write_text(json.dumps(relevance, indent=2), encoding="utf-8")
    print(f"{len(relevance)} questions, {len(passages)} passages -> {RELEVANCE_PATH}")


if __name__ == "__main__":
    main()