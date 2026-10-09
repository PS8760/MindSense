"""Face-module toggle: the env flag gates defaults, callers may override it."""

from __future__ import annotations

import pytest

from mindsense import inference


def test_face_gate_honours_env_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_FACE", "0")

    off = inference.analyze_face(b"x", enabled=False)
    assert off["available"] is False
    assert "switched off" in off["message"]

    default = inference.analyze_face(b"x")  # enabled=None → env says off
    assert default["available"] is False
    assert "switched off" in default["message"]

    # Explicit True (the app's session-enable button) must bypass the env flag.
    try:
        result = inference.analyze_face(b"x", enabled=True)
    except ValueError:
        result = {}
    assert "switched off" not in str(result.get("message", ""))
