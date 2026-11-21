# eval/

Evaluation of the RAG system on the dataset's questions. Each experiment runs on a fixed random sample of questions and saves per-question results and aggregated metrics as JSON.

| Path | Role |
|---|---|
| `run_eval.py` | Runs the system on a sample of questions, scores it, saves the results. |
| `build_relevance.py` | Counts, for each question, how many corpus passages contain its ground truth (needed for Recall@K). Run once. |
| `relevance.json` | Output of `build_relevance.py` (generated, not created until the script is run). |
| `results/` | One JSON file per experiment. |
| `metrics/retrieval.py` | Retrieval metrics. |
| `metrics/generation.py` | Generation metrics. |
| `metrics/operational.py` | Latency. |
| `metrics/text.py` | Text helpers shared by the metrics (not metrics). |

## Running

```bash
python eval/build_relevance.py                      # once
python eval/run_eval.py --name rag_baseline --n 50
```

| Option | Meaning | Default |
|---|---|---|
| `--name` | Experiment name (required), used as the output file name | none |
| `--n` | Number of questions | 100 |
| `--k` | Number of retrieved passages | `config.TOP_K` |
| `--seed` | Seed for the question sample | 42 |

The same seed gives the same sample, so different systems are compared on the same questions. A first unscored call warms up the model before measuring. Without `relevance.json`, the script prints a warning and Recall@K is skipped.

## Question types

The dataset mixes two kinds of questions, scored differently:

- **Yes/no questions** (ground truth is "yes" or "no"): scored with the yes/no accuracy only.
- **Open questions** (any other ground truth): scored with the retrieval metrics, exact match and token F1.

All the metrics below are computed per question, then averaged over the questions they apply to. The summary reports the mean and `n`, the number of questions the mean is based on.

## Text normalization

Several metrics compare texts after the same normalization: lowercase, accents removed, dashes turned into spaces, other punctuation deleted, articles (a, an, the) removed, spaces collapsed. Example: `"The Eiffel Tower."` becomes `"eiffel tower"`, and `"1,850"` becomes `"1850"`.

A text *contains* the ground truth if the normalized ground truth appears in the normalized text as whole words (`"1"` is not found in `"1850"`).

---

## Retrieval metrics

They answer one question: did the retriever bring back the right passages? Computed on open questions only, on the `k` passages retrieved for each question, in rank order.

**Proxy labels.** The questions are not linked to passages, so a passage is labelled *relevant* if it contains the ground truth (whole-word match). This is an approximation: a short answer can appear in an irrelevant passage, and a relevant passage may phrase the answer differently.

**Corpus count.** `build_relevance.py` scans all the passages of the dataset, while the index skips those shorter than `MIN_PASSAGE_CHARS` (see `src/README.md`). Passages are indexed as they are (no chunking), so the two use the same units, but the corpus count can include a few passages absent from the index.

### Precision@K

- **Measures:** how much of what was retrieved is useful (noise level).
- **Computation:** `relevant passages in the top-k / number of retrieved passages`.
- **Range:** 0 to 1. 1 means every retrieved passage is relevant; 0 means none is.
- **Reading it:** low values mean the retriever brings back a lot of noise. It tends to decrease as `k` grows, since extra passages are less likely to be relevant.

### MRR (Mean Reciprocal Rank)

- **Measures:** how high the first relevant passage is ranked.
- **Computation:** per question, the reciprocal rank is `1 / rank of the first relevant passage`, or 0 if none of the top-k is relevant. MRR is its mean over the open questions.
- **Range:** 0 to 1. 1 means the first passage is always relevant; 0.5 means on average the second; 0.33 the third; 0 means no relevant passage in the top-k.
- **Reading it:** it only looks at the first relevant passage, so it matters most when the model mostly relies on the top result. It assumes the retriever returns passages sorted by score.

### Recall@K

- **Measures:** how much of the relevant material in the corpus was found.
- **Computation:** `relevant passages in the top-k / relevant passages in the whole corpus`. The corpus count comes from `relevance.json`.
- **Range:** 0 to 1. 1 means every relevant passage of the corpus was retrieved; 0 means none.
- **Reading it:** the best reachable value is `min(k, N) / N`, where `N` is the number of relevant passages in the corpus, so a low recall is expected when many passages match and `k` is small. Questions with no relevant passage in the corpus (`N = 0`) are skipped, which is why `n` can be smaller than the number of open questions. A value above 1 means the corpus scan and the index do not use the same passages, and the script prints a warning.

---

## Generation metrics

They answer one question: does the generated answer match the ground truth? They look at the final answer only, not at the retrieved passages.

### Yes/no accuracy

- **Measures:** whether the system answers yes/no questions correctly.
- **Computation:** per question, 1 if the answer starts with the expected "yes" or "no" (after normalization), else 0. The accuracy is the mean over yes/no questions. Answers that start with a refusal such as "No relevant information found" are not parsed as "no".
- **Range:** 0 to 1. 1 means all answers are correct.
- **Reading it:** compare it with `yes_no_majority_baseline`, the score of a system that always answers the most frequent class. Also check `yes_no_unparsed_rate`, the share of answers where no yes/no could be read. They count as wrong in the accuracy, but it is a formatting problem rather than a wrong answer: a high value points to the prompt or the output format.

### Exact match

- **Measures:** whether the answer is identical to the ground truth.
- **Computation:** 1 if the normalized answer equals the normalized ground truth, else 0. Mean over open questions.
- **Range:** 0 to 1. 1 means every answer is identical to the ground truth.
- **Reading it:** the strictest metric. A correct but verbose answer ("Lincoln was born in 1809" for "1809") scores 0, so it mostly measures how short and direct the answers are.

### Token F1

- **Measures:** word overlap between the answer and the ground truth.
- **Computation:** with `overlap` the number of words shared by both (repeated words counted correctly), `precision = overlap / words in the answer`, `recall = overlap / words in the ground truth`, and `F1 = 2 × precision × recall / (precision + recall)`. 0 if no word is shared. Mean over open questions.
- **Range:** 0 to 1. 1 means the same words; 0 means no word in common.
- **Reading it:** more tolerant than exact match, but verbosity lowers precision: "Lincoln was born in 1809" against "1809" gives about 0.33 even though it is correct. It ignores meaning, so a paraphrase of the right answer can score low.

---

## Operational metrics

They answer one question: how long does an answer take?

### Latency

- **Measures:** time taken to answer one question, from the question to the final answer (retrieval and generation together).
- **Computation:** wall-clock time per question, summarized by the **median** (typical case) and the **p95** (the value 95% of the questions stay under, showing the slow cases). The mean is not used because a few slow calls would distort it. `retrieval_s` and `generation_s` give the same summary per stage, when `answer()` returns them.
- **Range:** seconds, 0 and above. Lower is better.
- **Reading it:** depends on the hardware, so compare runs on the same machine. A large gap between median and p95 means unstable response times.

---

## Output

`results/<name>.json` contains:

- `config`: models, `k`, sample size and seed.
- `summary`: aggregated metrics. Most are `{"mean": ..., "n": ...}`; latencies are `{"median": ..., "p95": ...}`.
- `rows`: one entry per question, with the question, ground truth, prediction and latency. Yes/no rows add `correct` and `parsed`. Open rows add `exact_match`, `f1`, `precision_at_k`, `reciprocal_rank`, `recall_at_k`, `n_relevant_in_corpus`, `hits` (one flag per retrieved passage, in rank order) and `contexts` (the retrieved texts).

Reading a few rows by hand is more informative than the averages.

## Limitations

- Retrieval metrics rely on proxy labels (see above).
- String-based metrics ignore meaning: a correct paraphrase can score 0.
- Small samples are noisy, and no confidence interval is computed: a small gap between two runs may be noise.
- The retrieval/generation latency split is only available if `answer()` returns `retrieval_s` and `generation_s`.
- Latency depends on the machine and on what else it is running.