"""
Role 6 – LLM wrapper (Groq Cloud API instead of Ollama)

Provides `ask_llm(question, context_chunks)` which:
  1. Builds a grounded system prompt instructing the model to answer
     ONLY from the supplied syllabus context.
  2. Sends the request to the Groq chat-completions endpoint.
  3. Returns the model's answer as a plain string.

Environment variables (loaded from backend/.env):
  GROQ_API_KEY  – your Groq API key
  GROQ_MODEL    – model id, default "llama-3.3-70b-versatile"
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# ── Load .env sitting next to this file ──────────────────────────────
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

_API_KEY: str = os.getenv("GROQ_API_KEY", "")
_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not _API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to backend/.env or export it as an environment variable."
    )

# ── Groq client (reused across requests) ─────────────────────────────
_client = Groq(api_key=_API_KEY)

# ── System prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a helpful teaching assistant. "
    "Answer the student's question using ONLY the syllabus excerpts "
    "provided below as context. "
    "If the answer cannot be found in the provided context, reply exactly: "
    '"I couldn\'t find that in the syllabus." '
    "Do not make up information."
)


def _build_user_message(question: str, context_chunks: list[str]) -> str:
    """Combine the retrieved chunks and the question into one user message."""
    context_block = "\n\n---\n\n".join(context_chunks)
    return (
        f"### Syllabus Context\n\n{context_block}\n\n"
        f"### Question\n\n{question}"
    )


def ask_llm(
    question: str,
    context_chunks: list[str],
    *,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Send a grounded question to the Groq LLM and return the answer.

    Parameters
    ----------
    question : str
        The student's question.
    context_chunks : list[str]
        Relevant syllabus chunks retrieved by the retriever (Role 5).
    temperature : float, optional
        Sampling temperature (lower = more deterministic). Default 0.3.
    max_tokens : int, optional
        Maximum tokens in the response. Default 1024.

    Returns
    -------
    str
        The model's answer, grounded in the supplied context.
    """
    if not context_chunks:
        return "I couldn't find that in the syllabus."

    user_message = _build_user_message(question, context_chunks)

    chat_completion = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return chat_completion.choices[0].message.content.strip()
