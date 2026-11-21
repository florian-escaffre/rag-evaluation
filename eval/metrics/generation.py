"""Generation metrics: does the generated answer match the ground truth?"""

from collections import Counter

from eval.metrics.text import normalize, parse_yes_no


# Yes/no accuracy: share of yes/no questions answered correctly | [0, 1] | 1 = all correct (compare with the majority-class baseline)
def yes_no_correct(prediction: str, ground_truth: str) -> bool:
    """Per question; the accuracy is the mean over yes/no questions.

    An unparsable answer counts as wrong here; its rate is reported separately.
    """
    return parse_yes_no(prediction) == normalize(ground_truth)


# Exact match: answer identical to the ground truth after normalization | {0, 1} per question | 1 = identical, 0 = any difference
def exact_match(prediction: str, ground_truth: str) -> bool:
    """Per question; the reported value is the mean over open questions."""
    return normalize(prediction) == normalize(ground_truth)


# Token F1: word overlap between answer and ground truth | [0, 1] | 1 = same words, 0 = no shared word (penalizes verbose answers)
def token_f1(prediction: str, ground_truth: str) -> float:
    """Per question; the reported value is the mean over open questions."""
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()

    # Counter intersection keeps the minimum count of each shared word
    overlap = sum((Counter(pred_tokens) & Counter(truth_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)