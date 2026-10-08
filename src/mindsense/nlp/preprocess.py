"""Deterministic text normalisation for social-media and journal text.

Kept intentionally rule-based (no model downloads) so cleaning is
reproducible and the same function is used by the pipeline, the training
notebooks and the deployed app's text check-in page.
"""

from __future__ import annotations

import html
import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"@\w{1,40}")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s!?.,;:'-]")

# A small emoji -> word map (frequent emotions); anything else emoji-like
# is stripped. Mapping (rather than deleting) keeps emotional signal.
_EMOJI_MAP = {
    "😭": " crying ",
    "😢": " cry ",
    "😞": " sad ",
    "😔": " sad ",
    "😟": " worried ",
    "😨": " scared ",
    "😰": " anxious ",
    "😩": " exhausted ",
    "😫": " tired ",
    "😤": " frustrated ",
    "😠": " angry ",
    "😡": " angry ",
    "🥺": " pleading ",
    "❤️": " love ",
    "❤": " love ",
    "😍": " love ",
    "😊": " happy ",
    "🙂": " smile ",
    "😀": " smile ",
    "😅": " relief ",
    "😂": " laughing ",
    "🥰": " cared for ",
    "👍": " thumbsup ",
    "👎": " thumbsdown ",
    "💪": " strong ",
    "✨": " sparkles ",
    "💕": " love ",
    "💔": " heartbreak ",
}
_EMOJI_RE = re.compile(
    "[" + "".join(map(re.escape, _EMOJI_MAP)) + r"]|[\U0001F300-\U0001FAFF☀-➿]"
)


def normalize_text(text: str, *, min_length: int = 0) -> str:
    """Normalise a raw text sample.

    Steps: HTML unescape -> unicode NFKC -> emoji to word tokens ->
    strip URLs and @handles -> lowercase -> keep word/punctuation chars ->
    collapse whitespace. Returns "" when the result is shorter than
    ``min_length`` characters.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _EMOJI_RE.sub(lambda m: _EMOJI_MAP.get(m.group(0), " "), text)
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" user ", text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    if min_length and len(text) < min_length:
        return ""
    return text


def word_count(text: str) -> int:
    """Number of whitespace-delimited tokens."""
    return len(text.split()) if text else 0
