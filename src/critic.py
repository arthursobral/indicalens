"""Critic/Verifier Agent: checks each claim in a generated answer against the
retrieved context with a local NLI model, so unsupported or contradicted
claims are flagged instead of reaching the user looking authoritative.

Same model as the Indicator Analyst (mDeBERTa-v3-base-mnli-xnli, multilingual
NLI). Prototyped before building: it cleanly separated a supported claim
(entailment 1.0), a wrong number (contradiction 1.0) and an unsupported
claim (neutral 1.0) at ~1.4s per (premise, claim) pair, and handled
"0,71 %" vs "0.71%" without any number normalisation on our side.
"""

import re
from functools import lru_cache

_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ENTAILMENT_THRESHOLD = 0.5

_REFUSAL_MARKERS = ("nao ha dados", "nao tenho dados", "nao ha dado", "nao ha informacao")


@lru_cache(maxsize=1)
def _nli():
    from transformers import pipeline
    return pipeline("text-classification", model=_MODEL, top_k=None)


def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def is_refusal(answer: str) -> bool:
    return any(m in _strip_accents(answer) for m in _REFUSAL_MARKERS)


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Returns [(claim_text, cited_chunk_numbers), ...]. Splits on lines and
    sentence ends, drops markdown emphasis, keeps the [n] citations aside
    (used to check the cited chunk first) and skips fragments too short to
    be a claim (headings, lone citations).
    """
    claims = []
    for line in re.split(r"[\n]+", answer):
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9*\-])", line):
            cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
            text = re.sub(r"\[\d+\]", "", sentence)
            text = re.sub(r"[*_`#>]", "", text).strip(" -•\t")
            if len(text.split()) >= 4 and not text.endswith(":"):  # ":" = lead-in, not a claim
                claims.append((text, cited))
    return claims


def _score(premise: str, claim: str) -> dict:
    out = _nli()({"text": premise, "text_pair": claim}, truncation=True, max_length=512)
    out = out[0] if isinstance(out[0], list) else out
    return {x["label"]: x["score"] for x in out}


def verify_claim(claim: str, cited: list[int], context: list[dict]) -> dict:
    """Checks the cited chunk(s) first (1 model pass when the LLM cited
    correctly), then falls back to every chunk. Supported if any chunk
    entails the claim; otherwise 'contradicted' if the best-matching chunk
    contradicts it, else 'unsupported'.
    """
    order = [n - 1 for n in cited if 1 <= n <= len(context)]
    order += [i for i in range(len(context)) if i not in order]

    best = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    for i in order:
        scores = _score(context[i]["content"], claim)
        if scores["entailment"] >= ENTAILMENT_THRESHOLD:
            return {"claim": claim, "status": "supported", "chunk": i + 1, "score": scores["entailment"]}
        if scores["contradiction"] > best["contradiction"]:
            best = scores
    status = "contradicted" if best["contradiction"] >= ENTAILMENT_THRESHOLD else "unsupported"
    return {"claim": claim, "status": status, "chunk": None, "score": best["contradiction"]}


def verify(answer: str, context: list[dict]) -> list[dict]:
    """One verdict per claim in `answer`. Sentences that admit a lack of
    data ('nao ha dados...') aren't claims to ground and are skipped, but
    the rest of a mixed answer is still checked.
    """
    return [
        verify_claim(text, cited, context)
        for text, cited in split_claims(answer)
        if not is_refusal(text)
    ]
