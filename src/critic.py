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


_NON_CLAIM_START = ("fonte", "se precisar", "se quiser", "caso queira", "caso precise")


def _table_row_to_text(line: str) -> str | None:
    """'| Nordeste | 7,6 % |' -> 'Nordeste: 7,6 %'. Header rows (no digit in
    any cell) and separator rows are not claims and return None.
    """
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    cells = [c for c in cells if c and not set(c) <= set("-: ")]
    if not cells or not any(ch.isdigit() for c in cells for ch in c):
        return None
    return ": ".join(cells) if len(cells) == 2 else " - ".join(cells)


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Returns [(claim_text, cited_chunk_numbers), ...]. Splits on lines and
    sentence ends, drops markdown emphasis, keeps the [n] citations aside,
    turns table rows into 'label: value' and skips what is not a factual
    claim: table headers, 'Fonte:' lines, offers of more help, lead-ins
    ending in ':' and short heading-like fragments without any number.
    (Measured on 116 real answer sentences: those non-claims were noise in
    the faithfulness metric.)
    """
    claims = []
    for line in answer.splitlines():
        is_row = line.lstrip().startswith("|")
        row = _table_row_to_text(line) if is_row else line
        if row is None:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9*\-])", row):
            cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
            text = re.sub(r"\[\d+\]", "", sentence)
            text = re.sub(r"[*_`#>]", "", text).strip(" -•\t|")
            n_words = len(text.split())
            if text.lower().startswith(_NON_CLAIM_START) or text.endswith(":"):
                continue
            has_digit = any(ch.isdigit() for ch in text)
            # a table row already has a digit; prose needs 4+ words with a number or 8+ words
            if n_words >= 8 or (has_digit and n_words >= (2 if is_row else 4)):
                claims.append((text, cited))
    return claims


def _score(premise: str, claim: str) -> dict:
    out = _nli()({"text": premise, "text_pair": claim}, truncation=True, max_length=512)
    out = out[0] if isinstance(out[0], list) else out
    return {x["label"]: x["score"] for x in out}


def _premises(claim: str, context: list[dict], top: int = 4) -> list[tuple[int, str]]:
    """Candidate premises as (chunk_number, text). Short chunks (one-line
    table facts) are used whole; long prose chunks are split into sentences
    and only the `top` most similar to the claim are kept. A whole 700-char
    chunk with table captions in the middle confuses the NLI model: the eval
    harness measured false 'contradicted' on claims copied from the source.
    """
    candidates = []
    for i, chunk in enumerate(context, start=1):
        text = chunk["content"]
        if len(text) <= 300:
            candidates.append((i, text))
        else:
            candidates += [(i, sent) for sent in re.split(r"(?<=[.!?])\s+", text) if len(sent.split()) >= 4]
    if len(candidates) <= top:
        return candidates
    from src.embeddings import embed, embed_one

    claim_vec = embed_one(claim)
    sims = [sum(a * b for a, b in zip(claim_vec, vec)) for vec in embed([t for _, t in candidates])]
    ranked = sorted(zip(sims, candidates), key=lambda x: -x[0])
    return [c for _, c in ranked[:top]]


_STOPWORDS = {"que", "para", "com", "dos", "das", "nas", "nos", "por", "uma", "mais", "entre", "como", "foi", "foram", "era", "eram", "sao"}


def _numbers(text: str) -> set[str]:
    """Numbers as canonical strings ('7,6' and '7.60' -> '7.6'); years are
    left out because they are context, not the value being claimed."""
    text = re.sub(r"\b(?:19|20)\d{2}\b", "", text)
    out = set()
    for raw in re.findall(r"\d+(?:[.,]\d+)?", text):
        raw = raw.replace(",", ".")
        out.add(raw.rstrip("0").rstrip(".") if "." in raw else raw)
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", _strip_accents(text)) if w not in _STOPWORDS}


def is_short_numeric(claim: str) -> bool:
    """A short statement carrying a number ('Nordeste: 7,6 %', 'A taxa foi de
    5,4 %'). The mDeBERTa NLI flagged 18 of 62 source-supported claims in a
    hand-labeled set of real answers, many of them of this kind; checking
    numbers + label words lexically removed 6 of those 18 false alarms with
    no loss of recall (7/9). Requiring the 'label: value' format removed the
    gain, so it is short numeric claims in general, not table rows, that the
    NLI handles badly. Heuristic measured on a small set (see docs).
    """
    return len(claim.split()) <= 9 and bool(_numbers(claim))


def _short_numeric_supported(claim: str, context: list[dict], min_overlap: float = 0.7) -> bool:
    """All the numbers of the fragment and >= 70% of its label words appear
    together in one source sentence or chunk. Limit: it does not see WHICH
    column/year a number belongs to (a right number from the wrong quarter
    would pass), so it is only used for short numeric claims.
    """
    numbers, words = _numbers(claim), _content_words(claim)
    for chunk in context:
        text = chunk["content"]
        for piece in re.split(r"(?<=[.!?])\s+", text) + [text]:
            if numbers <= _numbers(piece) and len(words & _content_words(piece)) >= min_overlap * len(words):
                return True
    return False


def verify_claim(claim: str, cited: list[int], context: list[dict]) -> dict:
    """Short numeric claims are checked lexically (see is_short_numeric). Full sentences:
    supported if any candidate premise entails the claim (>= threshold);
    otherwise 'contradicted' if the best premise contradicts it, else
    'unsupported'.
    """
    if is_short_numeric(claim):
        ok = _short_numeric_supported(claim, context)
        return {"claim": claim, "status": "supported" if ok else "unsupported", "chunk": None, "score": 1.0 if ok else 0.0}
    best = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    for chunk_no, premise in _premises(claim, context):
        scores = _score(premise, claim)
        if scores["entailment"] >= ENTAILMENT_THRESHOLD:
            return {"claim": claim, "status": "supported", "chunk": chunk_no, "score": scores["entailment"]}
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
