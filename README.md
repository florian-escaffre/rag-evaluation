# rag-evaluation

A personal learning project to get hands-on with **Retrieval-Augmented Generation (RAG)** and, above all, with how to **evaluate** it, using **LangChain** with small local models served by **Ollama**.

This is not a production system. The goal is to understand each building block by implementing it, measuring it, and knowing where the errors come from: bad retrieval or bad use of good passages.

## Goals

- Understand the RAG principle: embeddings, vector search, prompting with retrieved context.
- Learn how to evaluate a RAG system, separating retrieval errors from generation errors.
- Go beyond string metrics: add an LLM-as-judge, and validate the judge itself on a manually labelled sample.
- Run everything locally on a small GPU with small models (about 3B parameters).

## How it works

```
INDEXING (once)                     QUERYING (each question)
src/ingest.py                       src/rag_chain.py -> src/retriever.py

passages -> embeddings -> Chroma    question -> embedding -> top-k passages -> LLM -> answer
```

1. **Indexing:** every passage of the corpus is turned into a vector by an embedding model and stored in a Chroma vector database.
2. **Retrieval:** the question is embedded with the same model, and the k closest passages are fetched.
3. **Generation:** the passages and the question are put in a prompt, and a local LLM answers using only that context (temperature=0, for reproducible evaluation).
4. **Evaluation:** a fixed random sample of questions is run through the system, and answers and retrieved passages are scored.

## Dataset

[`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) from Hugging Face:

- `text-corpus`: 3,200 short Wikipedia passages (about 390 characters on average).
- `question-answer`: 918 questions with short answers, many of them yes/no.

The questions are not linked to specific passages, so retrieval quality can only be estimated with a proxy (see [`eval/`](eval/README.md)). Check the dataset card for its license.

## Models

Served locally by Ollama and set in `config.py`:

| Role | Model |
|---|---|
| Embeddings | `nomic-embed-text` |
| Generation | a ~3B model (e.g. `ministral-3:3b`) |

## Project structure

```
rag-evaluation/
├── config.py          # models, paths and parameters in one place
├── requirements.txt
├── src/               # RAG pipeline: indexing, retrieval, generation (see src/README.md)
├── eval/              # evaluation code, experiment scripts and results (see eval/README.md)
└── db/                # Chroma index, generated locally (see db/README.md)
```

## Setup

Requirements: Python 3.10+, [Ollama](https://ollama.com) running locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull ministral-3:3b        # or the model set in config.py
```

## Usage

```bash
# 1. Inspect the dataset
python src/explore.py

# 2. Build the vector index (about a minute; passages under 20 characters are skipped)
python src/ingest.py

# 3. Ask a few test questions
python src/rag_chain.py

# 4. Evaluate on a sample of questions
python eval/build_relevance.py                      # once, needed for Recall@K
python eval/run_eval.py --name rag_baseline --n 50

# 5. Analyze the results (notebook, to be added)
```

## Evaluation methodology

Each experiment runs a fixed random sample of questions (same seed, so experiments are comparable) and saves full per-question results as JSON in `eval/results/`.

Current metrics, in three groups:

- **Retrieval** (open questions): Precision@K, MRR and Recall@K, computed against proxy relevance labels (a passage is relevant if it contains the ground truth).
- **Generation:** yes/no accuracy on yes/no questions, exact match and token-level F1 on open questions.
- **Operational:** latency (median and p95, total and per stage).

Comparing retrieval and generation metrics shows where errors come from: bad retrieval or bad use of good passages. Definitions, interpretation and limits are in [`eval/README.md`](eval/README.md).

Planned next steps:

- **Hyperparameter study:** vary retrieval parameters (e.g. `k`) and compare runs on the same seed.
- **LLM-as-judge:** score generated answers with a local LLM instead of string comparison, better suited to rephrased or verbose answers.
- **Judge validation:** check the judge against manual labels on a small sample before trusting its scores.

## Status

- [x] Dataset exploration
- [x] Indexing with Chroma and Ollama embeddings
- [x] Simple RAG pipeline (LangChain building blocks)
- [x] Evaluation script: retrieval (proxy labels), generation (string-based) and operational metrics
- [ ] Baseline run with the current metrics
- [ ] Hyperparameter study
- [ ] LLM-as-judge and manual validation of the judge

## Results

To be filled in once each experiment has been run and reviewed.

## Limitations

- Small models (about 3B) follow instructions less reliably than large ones.
- Retrieval quality is estimated through a proxy, not measured against labelled passages.
- String-based metrics measure surface similarity, not meaning: a correct but rephrased answer can be scored low.
- Small evaluation samples give noisy numbers; avoid reading much into small differences.

## Tech stack

Python, LangChain, Chroma, Ollama, Hugging Face `datasets`.
