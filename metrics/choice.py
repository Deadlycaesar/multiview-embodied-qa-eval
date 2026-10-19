"""Conservative extraction, separate from strict response-format compliance."""

import re


def format_compliant(text, choices):
    return text.strip() in choices


def normalized(text):
    return " ".join(text.casefold().strip().rstrip(".!?").split())


def extract_choice(text, choices, allow_exact_text=True):
    value = text.strip()
    candidates = set()
    labels = "|".join(re.escape(label) for label in sorted(choices, key=len, reverse=True))
    if not labels:
        return {"letter": None, "parse_status": "unparsed"}
    bare = re.fullmatch(r"\(?([A-Za-z]+)\)?[.)]?", value)
    if bare and bare.group(1).upper() in choices:
        return {"letter": bare.group(1).upper(), "parse_status": "letter"}
    # Explicit answer declarations anywhere, or a leading punctuated letter.
    declarations = re.findall(
        r"\b(?:answer|option|choice)(?:\s+is)?\s*[:=]?\s*\(?([A-Za-z]+)\)?"
        r"(?=\s*(?:$|[.,:;!?]|because\b|since\b))", value, re.I,
    )
    candidates.update(letter.upper() for letter in declarations)
    leading = re.match(rf"^\(?({labels})\)?[.:)]\s+", value, re.I)
    if leading:
        candidates.add(leading.group(1).upper())
    for pair in re.findall(rf"\b({labels})\s*(?:/|,|\bor\b|\band\b)\s*({labels})\b", value, re.I):
        candidates.update(letter.upper() for letter in pair)
    if candidates:
        # Multiple mentioned letter choices are ambiguous, even if one is called final.
        mentions = re.findall(r"\b([A-Z]+)\b", value)
        candidates.update(mentions)
        candidates.update(letter.upper() for letter in re.findall(r"\(([a-z]+)\)", value, re.I))
    exact = [letter for letter, choice in choices.items() if normalized(choice) == normalized(value)] if allow_exact_text else []
    candidates.update(exact)
    if len(candidates) > 1:
        return {"letter": None, "parse_status": "ambiguous"}
    if candidates and next(iter(candidates)) in choices:
        return {"letter": next(iter(candidates)), "parse_status": "choice_text" if exact else "explicit_letter"}
    return {"letter": None, "parse_status": "unparsed"}


def choice_letter(text, choices=None):
    """Legacy bare-label helper; scoring always supplies the actual choices."""
    if choices is not None:
        return extract_choice(text, choices, False)["letter"]
    match = re.fullmatch(r"\s*\(?([A-Za-z]+)\)?[.)]?\s*", text)
    return match.group(1).upper() if match else None
