"""LLM and embedding clients.

``DatabricksLLM`` calls a Databricks Model Serving / Foundation Model API endpoint through
the OpenAI-compatible client that ``WorkspaceClient`` hands out — no API keys in code; the
notebook's identity is used. ``EchoLLM`` is a deterministic stand-in for tests: it answers
by quoting the best-matching context sentence, so guardrail and citation logic can be
tested without a model.
"""
from __future__ import annotations

import re
from typing import Sequence

from .chunking import split_sentences, tokenize

_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "what", "how",
         "do", "does", "i", "we", "my", "our", "be", "with", "by", "at", "it", "that", "this", "when",
         "which", "who", "can", "should", "must", "if", "many", "much", "long", "within", "time", "policy"}


def _stem(tok: str) -> str:
    for suf in ("ations", "ation", "ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _terms(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if t not in _STOP}


class DatabricksLLM:
    def __init__(self, endpoint: str, temperature: float = 0.0, max_tokens: int = 512, client=None):
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is None:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        self.client = client

    def chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.endpoint, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens)
        return resp.choices[0].message.content or ""

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])


class DatabricksEmbedder:
    def __init__(self, endpoint: str = "databricks-gte-large-en", batch_size: int = 64, client=None):
        self.endpoint = endpoint
        self.batch_size = batch_size
        if client is None:
            from databricks.sdk import WorkspaceClient

            client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        self.client = client

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.endpoint, input=list(texts[i : i + self.batch_size]))
            out += [d.embedding for d in resp.data]
        return out


class EchoLLM:
    """Test double: returns the context sentence with the highest content-word overlap + its
    citation, or the refusal text when the context covers under half of the question."""

    def __init__(self, refusal: str = "I don't know based on the documents you have access to."):
        self.refusal = refusal
        self.last_messages: list[dict] = []

    def chat(self, messages: list[dict]) -> str:
        self.last_messages = messages
        user = messages[-1]["content"]
        q = _terms(re.search(r"Question:\s*(.*)", user, re.S).group(1)) if "Question:" in user else set()
        best, best_id, best_score = None, None, 0
        for doc_id, body in re.findall(r"<source id=\"([^\"]+)\"[^>]*>\n(.*?)\n</source>", user, re.S):
            for s in split_sentences(body):
                score = len(q & _terms(s))
                if score > best_score:
                    best, best_id, best_score = s, doc_id, score
        if best is None or best_score < max(2, 0.4 * len(q)):
            return self.refusal
        return f"{best} [source: {best_id}]"

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])
