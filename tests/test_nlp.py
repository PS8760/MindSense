"""Shared text normalisation used by every corpus cleaner."""

from __future__ import annotations

from mindsense.nlp.preprocess import normalize_text, word_count


def test_urls_and_handles_are_stripped():
    out = normalize_text("see https://example.com/x or www.site.com @someone help")
    assert "http" not in out and "www" not in out and "@" not in out
    assert "user" in out  # handles become a placeholder token
    assert "see" in out and "help" in out


def test_emoji_becomes_emotional_word_not_deleted():
    out = normalize_text("i feel 😭 today")
    assert "crying" in out
    out2 = normalize_text("so ❤️ this")
    assert "love" in out2


def test_unknown_emoji_is_removed_not_left_raw():
    out = normalize_text("wow 🌀🌀")
    assert "🌀" not in out


def test_html_entities_unescaped_and_lowercased():
    out = normalize_text("I &amp; YOU &lt;3")
    assert "&" not in out and "i" in out and "you" in out


def test_whitespace_collapsed():
    assert normalize_text("a \n\t  b") == "a b"


def test_empty_and_non_string_inputs():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text(None) == ""
    assert normalize_text(12345) == ""


def test_min_length_filter():
    assert normalize_text("hi", min_length=5) == ""
    assert normalize_text("hello there", min_length=5) == "hello there"


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0
    assert word_count("  spaced  out ") == 2
