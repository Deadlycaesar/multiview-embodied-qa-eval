"""Unicode word counts, casefolded; no article removal or stemming."""

import re
from collections import Counter


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def token_f1(prediction, references):
    predicted = Counter(tokens(prediction))
    if not predicted or not references:
        return 0.0
    best = 0.0
    for reference in references:
        expected = Counter(tokens(reference))
        overlap = sum((predicted & expected).values())
        denominator = sum(predicted.values()) + sum(expected.values())
        best = max(best, 2 * overlap / denominator if denominator else 0.0)
    return best
