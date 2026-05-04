from __future__ import annotations

import re
import unicodedata

import pyarabic.araby as araby
from num2words import num2words

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_APOSTROPHES = frozenset({"'", "\u2019"})
_STRIP_CATEGORIES = frozenset({"P", "S", "C"})

_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _numbers_to_words(text: str) -> str:
    """Convert Eastern then Western digits to Arabic words — mirrors training."""
    text = text.translate(_EASTERN_DIGITS)

    def _replace(match: re.Match) -> str:
        try:
            return " " + num2words(int(match.group()), lang="ar") + " "
        except Exception:
            return match.group()

    text = re.sub(r"\d+", _replace, text)
    return _WHITESPACE_RE.sub(" ", text)


def normalize_tts_text(text: str) -> str:
    # 1. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    # 2. PyArabic normalization — exact same order as training
    text = araby.strip_tashkeel(text)  # remove harakat
    text = araby.strip_tatweel(text)  # remove kashida ـ
    text = araby.normalize_alef(text)  # أ إ آ → ا
    text = araby.normalize_hamza(text)  # hamza forms
    text = araby.normalize_ligature(text)  # ligatures

    # 3. Teh Marbuta → Heh (model never saw ة during training)
    text = text.replace("ة", "ه")

    # 4. Numbers → Arabic words
    text = _numbers_to_words(text)

    # 5. Character-level filtering
    category = unicodedata.category
    spoken_chars: list[str] = []
    n = len(text)

    for i, char in enumerate(text):
        if char.isalpha() or char.isdigit():
            spoken_chars.append(char)
            continue

        if char == " " or category(char)[0] == "Z":
            spoken_chars.append(" ")
            continue

        # Mid-word apostrophe
        if (
            char in _WORD_APOSTROPHES
            and 0 < i < n - 1
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        ):
            spoken_chars.append(char)
            continue

        # Everything else → dropped

    result = "".join(spoken_chars).strip()
    return _WHITESPACE_RE.sub(" ", result)
