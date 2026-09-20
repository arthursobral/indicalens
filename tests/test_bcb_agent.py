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


def test_periods():
    import datetime as dt
    f = bcb_agent.parse_periods
    assert f("selic em 15/03/2022") == [("date", dt.date(2022, 3, 15))]
    assert f("selic em 1º de janeiro de 2022") == [("date", dt.date(2022, 1, 1))]
    assert f("dolar em novembro de 2022") == [("month", 2022, 11)]
    assert f("selic de 2021 e 2022") == [("year", 2021), ("year", 2022)]
    assert f("selic em marco de 2020 e dezembro de 2021") == [("month", 2020, 3), ("month", 2021, 12)]
    assert f("selic mais recente") == []


def test_lines():
    D = __import__("datetime").date
    rows = {432: [(D(2022, 1, 1), 9.25), (D(2022, 11, 30), 13.75), (D(2022, 12, 31), 13.75)],
            4390: [(D(2022, 11, 1), 1.02)],
            1: [(D(2022, 12, 1), 5.0), (D(2022, 12, 30), 5.2177)]}
    real = bcb_client.fetch_range
    bcb_client.fetch_range = lambda code, s, e: rows[code]
    try:
        y = bcb_agent.answer("selic de 2022")["answer"]
        assert "terminou o ano em 13,75%" in y and "entre 9,25% e 13,75%" in y
        m = bcb_agent.answer("selic de novembro de 2022")["answer"]
        assert "13,75% a.a." in m and "1,02%" in m
        d = bcb_agent.answer("dolar em dezembro de 2022")["answer"]
        assert "R$ 5,2177 em 30/12/2022" in d and "média R$ 5,1089" in d
        two = bcb_agent.answer("selic em 2021 e 2022")["answer"]
        assert len(two.splitlines()) == 2
    finally:
        bcb_client.fetch_range = real


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
        assert "informe o ano" in bcb_agent.answer("Qual era a Selic em marco?")["answer"].lower()
        assert "não existe" in bcb_agent.answer("Selic em 30/02/2022")["answer"]
        assert "futura" in bcb_agent.answer("Selic em dezembro de 2099")["answer"]
        assert bcb_agent.answer("A inflacao se move junto com a Selic?") is None  # relations are the Correlation Agent's
        assert bcb_agent.answer("Qual foi o IPCA mais recente?") is None
    finally:
        bcb_client.fetch_latest = real


if __name__ == "__main__":
    test_latest_ignores_future_dated_rows()
    test_answers()
    test_periods()
    test_lines()
    print("test_bcb_agent: ok")
    if "--live" in sys.argv:
        print(bcb_agent.answer("qual o cambio mais recente para dolar de hoje e qual a taxa selic mais recente?")["answer"])
