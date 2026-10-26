"""The four guardrail layers of the assistant.

  1. Input filtration   — block prompt injection / jailbreak attempts before anything runs.
  2. PII masking        — redact personal data in retrieved context *before* it reaches the
                          LLM (and in the question and the answer).
  3. RBAC               — see ``rbac.py``; enforced as a retrieval filter.
  4. Output verification— score the answer's groundedness against the retrieved context and
                          check its citations; refuse instead of returning a hallucination.

Layers 1, 2 and 4 are deterministic and cheap, so they run on every request. They can be
complemented by model-based checks (an LLM judge for groundedness, Presidio or a
Databricks AI Guardrails endpoint for PII) — the interfaces are kept simple so those can
be swapped in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import split_sentences, tokenize

# --------------------------------------------------------------------- 1. input filtration
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|earlier|system)\b.{0,20}\b(instructions?|rules?|prompts?|guardrails?)\b", re.I)),
    ("reveal_system_prompt", re.compile(r"\b(reveal|show|print|repeat|leak|output)\b.{0,30}\b(system|hidden|initial)\s+(prompt|instructions?|message)\b", re.I)),
    ("role_hijack", re.compile(r"\b(you are now|act as|pretend (to be|you are)|from now on you)\b.{0,60}\b(unfiltered|unrestricted|no (rules|limits|restrictions)|jailbroken|dan|developer mode)\b", re.I)),
    ("jailbreak_keyword", re.compile(r"\b(jailbreak|do anything now|developer mode enabled|god mode)\b", re.I)),
    ("access_escalation", re.compile(r"\b(bypass|disable|turn off|skip)\b.{0,30}\b(access control|rbac|permissions?|filters?|guardrails?|security)\b", re.I)),
    ("exfiltration", re.compile(r"\b(send|post|upload|exfiltrate)\b.{0,40}\b(https?://|webhook|external (server|url))", re.I)),
    ("encoded_payload", re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")),
]

MAX_QUESTION_CHARS = 2000


@dataclass
class InputCheck:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


def check_input(question: str) -> InputCheck:
    reasons = []
    if not question or not question.strip():
        reasons.append("empty_question")
    if len(question) > MAX_QUESTION_CHARS:
        reasons.append("too_long")
    reasons += [name for name, pat in INJECTION_PATTERNS if pat.search(question or "")]
    return InputCheck(allowed=not reasons, reasons=reasons)


# --------------------------------------------------------------------------- 2. PII masking
def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums))
    return total % 10 == 0


PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-])?\d{3}[ .-]\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("ACCOUNT_NUMBER", re.compile(r"\b(?:acct|account)(?:\s*(?:no\.?|number|#))?\s*[:#]?\s*\d{6,17}\b", re.I)),
    ("DATE_OF_BIRTH", re.compile(r"\b(?:dob|date of birth)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)),
]


@dataclass
class Redaction:
    text: str
    findings: dict[str, int] = field(default_factory=dict)

    @property
    def found_pii(self) -> bool:
        return bool(self.findings)


def mask_pii(text: str) -> Redaction:
    findings: dict[str, int] = {}

    def make_sub(label: str):
        def _sub(m: re.Match) -> str:
            if label == "CREDIT_CARD":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return m.group()
            findings[label] = findings.get(label, 0) + 1
            return f"[{label}]"
        return _sub

    for label, pat in PII_PATTERNS:
        text = pat.sub(make_sub(label), text)
    return Redaction(text=text, findings=findings)


# --------------------------------------------------------------- 4. output verification
_STOP = set("""a an the and or of to in on for with by at from is are was were be been being this that
these those it its as if then than so such into within must may can will should not no any all each
per every their there they he she we you your our i""".split())
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_CITE = re.compile(r"\[(?:source|sources|doc)s?:\s*([^\]]+)\]", re.I)


def extract_citations(answer: str) -> list[str]:
    ids = []
    for group in _CITE.findall(answer):
        ids += [x.strip() for x in re.split(r"[,;\s]+", group) if x.strip()]
    return ids


def strip_citations(answer: str) -> str:
    return _CITE.sub("", answer).strip()


def sentence_support(sentence: str, context_tokens: set[str], context_numbers: set[str]) -> float:
    toks = [t for t in tokenize(sentence) if t not in _STOP]
    if not toks:
        return 1.0
    numbers = set(_NUM.findall(sentence))
    if numbers and not numbers <= context_numbers:
        return 0.0                        # an invented number is a hallucination, full stop
    return sum(t in context_tokens for t in toks) / len(toks)


@dataclass
class Verification:
    groundedness: float
    unsupported_sentences: list[str]
    cited_ids: list[str]
    invalid_citations: list[str]
    passed: bool


def verify_answer(answer: str, contexts: list[dict], min_groundedness: float = 0.55,
                  sentence_threshold: float = 0.5) -> Verification:
    """Lexical groundedness: share of answer sentences whose content words (and every
    number) are found in the retrieved context."""
    body = strip_citations(answer)
    ctx_text = " ".join(c.get("text", "") + " " + c.get("content", "") for c in contexts)
    ctx_tokens = set(tokenize(ctx_text))
    ctx_numbers = set(_NUM.findall(ctx_text))
    sentences = split_sentences(body) or ([body] if body else [])
    scores = [sentence_support(s, ctx_tokens, ctx_numbers) for s in sentences]
    unsupported = [s for s, sc in zip(sentences, scores) if sc < sentence_threshold]
    groundedness = (sum(sc >= sentence_threshold for sc in scores) / len(scores)) if scores else 0.0
    cited = extract_citations(answer)
    allowed = {c["doc_id"] for c in contexts}
    invalid = [c for c in cited if c not in allowed]
    passed = groundedness >= min_groundedness and not invalid and bool(cited)
    return Verification(round(groundedness, 3), unsupported, cited, invalid, passed)
