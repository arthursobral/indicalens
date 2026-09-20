"""Self-check for the batch job with a stubbed graph (hermetic; runs in CI).
Run: python tests/test_batch.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import batch, llm


def _state(answer, score=1.0, verdicts=()):
    return {"answer": answer, "context": [{"citation": "fonte X", "score": score}], "verdicts": list(verdicts)}


def test_isolates_failures_and_counts_tokens():
    calls = []

    def fake_ask(q):
        calls.append(q)
        if "quebra" in q:
            raise RuntimeError("429 too many requests")
        if "rag" in q:
            llm._count(100, 20)
            return _state("resposta", 0.5, [{"status": "supported"}, {"status": "unsupported"}, {"status": "inferred"}])
        return _state("valor")

    old = batch.REPORTS
    batch.REPORTS = {"A": ["direta", "rag", "quebra"], "B": ["direta"]}
    try:
        with tempfile.TemporaryDirectory() as d:
            s = batch.run(None, Path(d), fake_ask, lambda s: None)
            assert (Path(d) / "A.md").exists() and (Path(d) / "summary.json").exists()
            text = (Path(d) / "A.md").read_text(encoding="utf-8")
    finally:
        batch.REPORTS = old
    a, b = s["per_report"]
    assert s["reports"] == 2 and s["errors"] == 1  # the failure is recorded, B still ran
    assert a["llm_calls"] == 1 and a["tokens_in"] == 100 and a["tokens_out"] == 20  # tokens are per report, not cumulative
    assert b["llm_calls"] == 0 and b["tokens_in"] == 0
    assert a["flagged"] == 1 and a["claims"] == 3  # only hard flags count; "inferred" does not
    assert calls.count("quebra") == batch.RETRIES  # retried before giving up
    assert "**Erro:** RuntimeError" in text and "1 de 3 afirmações" in text


def test_path_detection():
    d = batch.ask_once("q", lambda q: _state("v", 1.0))
    r = batch.ask_once("q", lambda q: _state("v", 0.6))
    assert d["path"] == "deterministic" and r["path"] == "rag"


if __name__ == "__main__":
    test_isolates_failures_and_counts_tokens()
    test_path_detection()
    print("test_batch: ok")
