from __future__ import annotations

import re
import unicodedata

import pyarabic.araby as araby
from num2words import num2words

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_APOSTROPHES = frozenset({"'", "\u2019"})
_STRIP_CATEGORIES = frozenset({"P", "S", "C"})
_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_SENTENCE_ENDERS = frozenset({".", "!", "?"})
_ARABIC_TO_LATIN_PUNCT = str.maketrans("؟،؛", "? ؛")


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


def _fix_punctuation_spacing(text: str) -> str:
    """
    Normalize space around sentence-ending punctuation.
    Periods are fully removed — mid-sentence periods cause XTTS to stop early,
    and terminal periods are replaced with ! by _ensure_terminal_punctuation.
    """
    # Remove ALL periods — mid-sentence ones cause premature stops in XTTS
    text = text.replace(".", " ")

    # Collapse multiple spaces that follow punctuation
    text = re.sub(r"([!?])\s{2,}", r"\1 ", text)
    # Remove space BEFORE punctuation
    text = re.sub(r"\s+([!?])", r"\1", text)
    # Ensure exactly one space AFTER punctuation when followed by more text
    text = re.sub(r"([!?])(\S)", r"\1 \2", text)
    return text


def _ensure_terminal_punctuation(text: str) -> str:
    """
    XTTS GPT needs a sentence-ending token to predict EOS cleanly.
    '!' is preferred over '.' — community reports confirm '.' is a known
    hallucination trigger, while '!' produces cleaner stop-token prediction.
    """
    if not text:
        return text
    if text[-1] == ".":
        # Replace trailing period with ! — safer for XTTS stop-token prediction
        text = text[:-1] + "!"
    elif text[-1] not in {"!", "?"}:
        text += "!"
    return text


def normalize_tts_text(text: str) -> str:
    # 1. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    # 2. PyArabic normalization — exact same order as training
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    text = araby.normalize_ligature(text)

    # 3. Teh Marbuta → Heh
    text = text.replace("ة", "ه")

    # 4. Numbers → Arabic words
    text = _numbers_to_words(text)

    # 5. Normalize punctuation spacing before character filtering
    text = _fix_punctuation_spacing(text)

    # Map Arabic punctuation to Latin equivalents so they survive the filter
    text = text.translate(_ARABIC_TO_LATIN_PUNCT)

    # 6. Character-level filtering
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

        if (
            char in _WORD_APOSTROPHES
            and 0 < i < n - 1
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        ):
            spoken_chars.append(char)
            continue

        if char in _SENTENCE_ENDERS:
            spoken_chars.append(char)
            continue

    result = "".join(spoken_chars).strip()
    result = _WHITESPACE_RE.sub(" ", result)

    # 7. Guarantee terminal ! after all filtering
    result = _ensure_terminal_punctuation(result)

    return result


if __name__ == "__main__":
    test_sentence = "أهلاً بيك أ. فهد، مستشارك العقاري. من فضلك، ما هو اسم حضرتك؟ وهل ممكن أعرف رقم التواصل الخاص بحضرتك عشان نسهل علينا التنسيق بعد كده؟ وبعدها أقدر أسألك شوية أسئلة. إيه اللي حضرتك بتبحث عنه؟ هل بتبحث عن شيء معين؟ أو ودك أشوف لك العروض والمخطط العام للمشروع؟"
    print("Original :", test_sentence)
    print("Normalized:", normalize_tts_text(test_sentence))
