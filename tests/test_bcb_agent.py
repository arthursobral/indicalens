"""Self-check for the BCB lookup (stubbed HTTP, hermetic; runs in CI).
Run: python tests/test_bcb_agent.py [--live]
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bcb_agent, bcb_client


def _stub(latest):
    bcb_client.fetch_latest = lambda code, days=45: latest[code]


def test_latest_ignores_future_dated_rows():
    today = date.today()
    f = lambda d: d.strftime("%d/%m/%Y")
    rows = [{"data": f(today - timedelta(days=1)), "valor": "13.75"}, {"data": f(today), "valor": "13.75"}]
    real = bcb_client._rows
    seen = {}
    bcb_client._rows = lambda code, start, end: seen.update(end=end) or rows
    try:
        assert bcb_client.fetch_latest(432) == (f(today), 13.75)
        assert seen["end"] == f(today)  # request is bounded by today, so projections are never returned
    finally:
        bcb_client._rows = real


def test_answers():
    real = bcb_client.fetch_latest
    try:
        _stub({432: ("20/09/2026", 13.75), 1: ("18/09/2026", 5.1575)})
        a = bcb_agent.answer("qual a taxa selic mais recente?")["answer"]
        assert "13,75% a.a." in a and "20/09/2026" in a
        b = bcb_agent.answer("qual o cambio mais recente para dolar de hoje e qual a taxa selic mais recente?")["answer"]
        assert "R$ 5,1575" in b and "13,75%" in b  # both parts answered
        c = bcb_agent.answer("Qual a Selic e o IPCA mais recentes?")["answer"]
        assert "13,75%" in c and "separadamente" in c  # never silently drops the IBGE part
        assert "período específico" in bcb_agent.answer("Qual era a Selic em marco de 2020?")["answer"]
        assert bcb_agent.answer("A inflacao se move junto com a Selic?") is None  # relations are the Correlation Agent's
        assert bcb_agent.answer("Qual foi o IPCA mais recente?") is None
    finally:
        bcb_client.fetch_latest = real


if __name__ == "__main__":
    test_latest_ignores_future_dated_rows()
    test_answers()
    print("test_bcb_agent: ok")
    if "--live" in sys.argv:
        print(bcb_agent.answer("qual o cambio mais recente para dolar de hoje e qual a taxa selic mais recente?")["answer"])
