"""Minimal client for the Banco Central SGS API (api.bcb.gov.br), no API key.

Monthly series only (that is all the Correlation Agent needs). Returns the same
shape as ibge_client.fetch_series-style points: [("YYYYMM", float), ...].
"""

import time
from datetime import date, timedelta

import requests

BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

SERIES = {
    "SELIC_MENSAL": {"code": 4390, "label": "Taxa Selic acumulada no mês", "unit": "% a.m."},
    "SELIC_META": {"code": 432, "label": "Meta da taxa Selic definida pelo Copom", "unit": "% a.a."},
    "CAMBIO_USD_DIARIO": {"code": 1, "label": "Taxa de câmbio livre, dólar americano (venda), diário", "unit": "R$/US$"},
    "CAMBIO_USD_MEDIA_MENSAL": {"code": 3695, "label": "Taxa de câmbio livre, dólar americano (venda), média mensal", "unit": "R$/US$"},
}


def _rows(code: int, start: str, end: str) -> list[dict]:
    params = {"formato": "json", "dataInicial": start, "dataFinal": end}
    for attempt in range(3):  # SGS occasionally answers 5xx/HTML; retry before giving up
        try:
            resp = requests.get(BASE_URL.format(code=code), params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_latest(code: int, days: int = 45) -> tuple[str, float]:
    """Latest daily point dated today or earlier, as ("dd/mm/yyyy", value). Bounded by
    dataFinal=today on purpose: the Selic target series (432) carries the current
    target forward to FUTURE dates, so "the last row" would be a projection."""
    today = date.today()
    rows = _rows(code, (today - timedelta(days=days)).strftime("%d/%m/%Y"), today.strftime("%d/%m/%Y"))
    rows = [r for r in rows if r["valor"] not in ("", None)]
    if not rows:
        raise ValueError(f"SGS {code}: no data in the last {days} days")
    return rows[-1]["data"], float(rows[-1]["valor"])


def fetch_series(code: int, start: str = "01/01/1995", end: str | None = None) -> list[tuple[str, float]]:
    rows = _rows(code, start, end or date.today().strftime("%d/%m/%Y"))
    # SGS monthly points are dated the 1st of the month: "dd/mm/yyyy"
    return [(f"{r['data'][6:10]}{r['data'][3:5]}", float(r["valor"])) for r in rows if r["valor"] not in ("", None)]
