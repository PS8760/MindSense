"""Minimal Groq chat client shared by the inference facade and the app.

The app can run its "model" features with **no locally trained artifacts** by
calling Groq's OpenAI-compatible chat API with ``GROQ_API_KEY`` (from the
environment or the git-ignored ``.env``). This module is deliberately
torch-free and streamlit-free so ``mindsense.inference`` can depend on it.

Set ``MINDSENSE_DISABLE_GROQ=1`` to force the offline/local-only behaviour
(used by the test-suite so no test ever makes a network call).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
#: gpt-oss models are *reasoning* models — they emit a ``reasoning`` field
#: before ``content``; we give a generous token budget and fall back to the
#: reasoning text if the content is empty (see ``chat``).
MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def groq_key() -> str | None:
    """Resolve the Groq API key from the environment or the repo ``.env``."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        from dotenv import load_dotenv

        from mindsense.utils.io import repo_path

        env_path = repo_path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return os.environ.get("GROQ_API_KEY", "").strip() or None
    except Exception:  # noqa: BLE001 - key discovery must never crash the app
        return None
    return None


def available() -> bool:
    """Whether a Groq key is configured (ignoring the test kill-switch)."""
    return groq_key() is not None


def enabled() -> bool:
    """Whether the Groq backend should be used for inference.

    Respects ``MINDSENSE_DISABLE_GROQ=1`` so the test-suite stays offline.
    """
    if os.environ.get("MINDSENSE_DISABLE_GROQ") == "1":
        return False
    return available()


def chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.6,
    max_tokens: int = 1024,
    key: str | None = None,
) -> str:
    """One Groq chat-completions call; tries both models before raising."""
    key = (key or groq_key() or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not configured")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload_base = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error = "groq request failed"
    for model in MODELS:
        try:
            response = requests.post(
                GROQ_URL, headers=headers, json={**payload_base, "model": model}, timeout=25
            )
        except requests.RequestException as exc:  # network down -> caller falls back
            raise RuntimeError(f"network error: {exc}") from exc
        if response.status_code == 200:
            message = response.json()["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if not content:  # reasoning model spent its budget before the answer
                content = (message.get("reasoning") or "").strip()
            if content:
                return content
        last_error = f"HTTP {response.status_code}: {response.text[:180]}"
        if response.status_code in (400, 404):  # unknown model -> try the other
            continue
        break
    raise RuntimeError(last_error)


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    key: str | None = None,
) -> dict[str, Any]:
    """Like :func:`chat` but parse a JSON object from the reply.

    Tolerates ```json fences and any prose around the object; raises
    ``ValueError`` when no JSON object can be recovered.
    """
    text = chat(system, user, temperature=temperature, max_tokens=max_tokens, key=key)
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed reply
            raise ValueError(f"model did not return valid JSON: {exc}") from exc
    raise ValueError("model did not return valid JSON")
