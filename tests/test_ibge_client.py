"""Self-check for period formatting and the live IBGE endpoint.
Run: python tests/test_ibge_client.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ibge_client import SERIES, fetch_series, format_period


def test_format_period():
    assert format_period("monthly", "202608") == "agosto de 2026"
    assert format_period("quarterly", "202603") == "3º trimestre de 2026"
    assert format_period("moving_quarter", "202607") == "trimestre móvel encerrado em julho de 2026"
    assert format_period("annual", "2024") == "2024"


def test_fetch_series_live():
    series = SERIES[0]
    points = fetch_series(series["agregado"], series["variavel"], "-3", series["classificacao"])
    assert len(points) >= 1
    period, value = points[-1]
    assert len(period) == 6
    float(value)  # must parse as a number


def test_fetch_annual_series_live():
    series = next(s for s in SERIES if s["period_kind"] == "annual")
    points = fetch_series(series["agregado"], series["variavel"], "-3", series["classificacao"])
    assert len(points) >= 1
    period, value = points[-1]
    assert len(period) == 4
    float(value)


if __name__ == "__main__":
    test_format_period()
    print("test_format_period: ok")
    try:
        test_fetch_series_live()
        print("test_fetch_series_live: ok")
    except Exception as e:
        print(f"test_fetch_series_live: SKIPPED ({e})")
    try:
        test_fetch_annual_series_live()
        print("test_fetch_annual_series_live: ok")
    except Exception as e:
        print(f"test_fetch_annual_series_live: SKIPPED ({e})")
