"""Text helpers shared by the metrics. These are not metrics themselves."""

import re
import unicodedata

# Words that, right after "yes"/"no", signal a refusal rather than an answer
# (e.g. "No relevant information found in the context").
# Heuristic: check the unparsed rate and a few rows by hand.
REFUSAL_WORDS = {
    "information", "relevant", "context", "mention", "mentioned",
    "data", "evidence", "answer", "passage", "passages", "details",
}


def normalize(text: str) -> str:
    """Lowercase, strip accents, punctuation and articles, collapse spaces.

    Dashes become spaces ("twenty-one" -> "twenty one"); other punctuation
    is deleted ("1,850" -> "1850", "U.S." -> "us").
    e.g. "The Eiffel Tower." -> "eiffel tower".
    """
    text = unicodedata.normalize("NFKD", text.lower())
    chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Mn":              # combining accent mark: drop it
            continue
        if category == "Pd":              # dash or hyphen: split the words
            chars.append(" ")
        elif category.startswith("P"):    # any other punctuation: delete it
            continue
        else:
            chars.append(ch)
    text = "".join(chars)
    # Remove articles as whole words (\b = word boundary)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains(text: str, ground_truth: str) -> bool:
    """True if the normalized ground truth appears in the text as whole words.

    Padding both sides with spaces forces word boundaries, so "1" is
    no longer found in "1850".
    """
    truth = normalize(ground_truth)
    if not truth:  # an empty ground truth must not match everything
        return False
    return f" {truth} " in f" {normalize(text)} "


def parse_yes_no(text: str) -> str | None:
    """Return 'yes' or 'no' if the answer starts with one of them, else None.

    Refusals such as "No relevant information found" are rejected (None).
    None is a parsing failure, reported separately from wrong answers.
    """
    words = normalize(text).split()
    if not words or words[0] not in {"yes", "no"}:
        return None
    if len(words) > 1 and words[1] in REFUSAL_WORDS:
        return None
    return words[0]