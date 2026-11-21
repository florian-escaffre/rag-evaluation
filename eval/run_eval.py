import argparse
import json
import random
import sys
import time
from pathlib import Path

# Make the project root importable (for `config`, `eval`, `src`)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm

import config
from eval.metrics import (
    exact_match,
    latency_summary,
    normalize,
    parse_yes_no,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_hits,
    token_f1,
    yes_no_correct,
)
from src.rag_chain import answer

# Folder where each experiment is saved as a JSON file
RESULTS_DIR = config.ROOT / "eval" / "results"
# Proxy relevance labels, produced by eval/build_relevance.py
RELEVANCE_PATH = config.ROOT / "eval" / "relevance.json"


def mean(values: list[float]) -> float | None:
    """Average of a list; None if empty. Booleans count as 1 / 0."""
    return sum(values) / len(values) if values else None


def stat(values: list) -> dict:
    """Mean of the non-missing values, and how many values it is based on.

    n matters: Recall@K skips the questions with no relevant passage in the corpus.
    """
    values = [float(v) for v in values if v is not None]
    return {"mean": mean(values), "n": len(values)}


def main() -> None:
    # --- Command-line arguments: one script, many experiments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="experiment name")
    parser.add_argument("--n", type=int, default=100, help="number of questions")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Select the questions ---
    questions = list(load_dataset(config.DATASET_NAME, "question-answer")["test"])
    # Fixed seed: same random sample at every run, so experiments are comparable
    random.Random(args.seed).shuffle(questions)
    questions = questions[: args.n]

    # --- Proxy relevance labels (needed for Recall@K only) ---
    if RELEVANCE_PATH.exists():
        relevance = json.loads(RELEVANCE_PATH.read_text(encoding="utf-8"))
    else:
        relevance = {}
        print(f"WARNING: {RELEVANCE_PATH} not found, Recall@K will be skipped. "
              "Run eval/build_relevance.py first.")

    # --- Warm-up call (not scored): the first call pays the model loading cost ---
    answer(questions[0]["question"], k=args.k)

    # --- Run the RAG system on each question ---
    rows = []
    for item in tqdm(questions):
        ground_truth = item["answer"]

        # Measure end-to-end time (retrieval + generation)
        start = time.perf_counter()
        result = answer(item["question"], k=args.k)
        latency = time.perf_counter() - start

        row = {
            "question": item["question"],
            "ground_truth": ground_truth,
            "prediction": result["answer"],
            # Operational metrics
            "latency_s": round(latency, 3),
            # Optional split: only filled if answer() returns these keys
            "retrieval_s": result.get("retrieval_s"),
            "generation_s": result.get("generation_s"),
            # Yes/no questions are scored differently from open questions
            "is_yes_no": normalize(ground_truth) in {"yes", "no"},
        }

        if row["is_yes_no"]:
            row["correct"] = yes_no_correct(result["answer"], ground_truth)
            # None = parsing failure, tracked apart from wrong answers
            row["parsed"] = parse_yes_no(result["answer"])
        else:
            # Generation metrics
            row["exact_match"] = exact_match(result["answer"], ground_truth)
            row["f1"] = token_f1(result["answer"], ground_truth)

            # Retrieval metrics: one flag per retrieved passage, in rank order
            passages = [doc.page_content for doc in result["documents"]]
            hits = retrieval_hits(passages, ground_truth)
            # Keep the retrieved texts: the LLM judge (phase 2) needs them
            # and can then run offline on saved results
            row["contexts"] = passages
            row["hits"] = hits
            row["precision_at_k"] = precision_at_k(hits)
            row["reciprocal_rank"] = reciprocal_rank(hits)
            n_relevant = relevance.get(item["question"])
            row["n_relevant_in_corpus"] = n_relevant
            row["recall_at_k"] = recall_at_k(hits, n_relevant)
        rows.append(row)

    # Recall above 1 means the scanned units differ from the indexed ones
    if any((r.get("recall_at_k") or 0) > 1 for r in rows):
        print("WARNING: Recall@K > 1 for some questions. The corpus scan and the "
              "index do not use the same passages (chunking?).")

    # --- Aggregate the results ---
    yes_no = [r for r in rows if r["is_yes_no"]]
    open_q = [r for r in rows if not r["is_yes_no"]]

    # Share of the most frequent class: the score of a model that always says the same thing
    n_yes = sum(normalize(r["ground_truth"]) == "yes" for r in yes_no)
    majority = max(n_yes, len(yes_no) - n_yes) / len(yes_no) if yes_no else None

    summary = {
        "n_total": len(rows),
        "n_yes_no": len(yes_no),
        "n_open": len(open_q),
        # Retrieval (proxy labels, open questions)
        "open_precision_at_k": stat([r["precision_at_k"] for r in open_q]),
        "open_mrr": stat([r["reciprocal_rank"] for r in open_q]),
        "open_recall_at_k": stat([r["recall_at_k"] for r in open_q]),
        # Generation
        "yes_no_accuracy": stat([r["correct"] for r in yes_no]),
        "yes_no_unparsed_rate": stat([r["parsed"] is None for r in yes_no]),
        "yes_no_majority_baseline": majority,
        "open_exact_match": stat([r["exact_match"] for r in open_q]),
        "open_f1": stat([r["f1"] for r in open_q]),
        # Operational
        "latency_s": latency_summary([r["latency_s"] for r in rows]),
        "retrieval_s": latency_summary([r["retrieval_s"] for r in rows]),
        "generation_s": latency_summary([r["generation_s"] for r in rows]),
    }

    # --- Save everything: settings, summary and per-question details ---
    output = {
        "config": {
            "name": args.name,
            "llm": config.LLM_MODEL,
            "embedding": config.EMBEDDING_MODEL,
            "k": args.k,
            "n": args.n,
            "seed": args.seed,
        },
        "summary": summary,
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.name}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved to", path)


if __name__ == "__main__":
    main()
