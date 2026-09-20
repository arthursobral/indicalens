"""Minimal client for the Banco Central SGS API (api.bcb.gov.br), no API key.

Monthly series only (that is all the Correlation Agent needs). Returns the same
shape as ibge_client.fetch_series-style points: [("YYYYMM", float), ...].
"""

import time
from datetime import date

import requests

BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

SERIES = {
    "SELIC_MENSAL": {"code": 4390, "label": "Taxa Selic acumulada no mês", "unit": "% a.m."},
    "CAMBIO_USD_MEDIA_MENSAL": {"code": 3695, "label": "Taxa de câmbio livre, dólar americano (venda), média mensal", "unit": "R$/US$"},
}


def fetch_series(code: int, start: str = "01/01/1995", end: str | None = None) -> list[tuple[str, float]]:
    end = end or date.today().strftime("%d/%m/%Y")
    params = {"formato": "json", "dataInicial": start, "dataFinal": end}
    for attempt in range(3):  # SGS occasionally answers 5xx/HTML; retry before giving up
        try:
            resp = requests.get(BASE_URL.format(code=code), params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    # SGS monthly points are dated the 1st of the month: "dd/mm/yyyy"
    return [(f"{r['data'][6:10]}{r['data'][3:5]}", float(r["valor"])) for r in rows if r["valor"] not in ("", None)]
