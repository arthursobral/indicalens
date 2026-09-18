"""Minimal client for the IBGE aggregated-data API (SIDRA), no API key required.

Uses servicodados.ibge.gov.br/api/v3/agregados instead of apisidra.ibge.gov.br
because the latter sits behind a Cloudflare bot challenge that blocks plain
HTTP clients; the v3 endpoint returns the same underlying SIDRA data as JSON.
"""

import requests

BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"

_MISSING_VALUES = {"...", "-", "X", ".."}

MONTHS_PT = {
    "01": "janeiro", "02": "fevereiro", "03": "março", "04": "abril",
    "05": "maio", "06": "junho", "07": "julho", "08": "agosto",
    "09": "setembro", "10": "outubro", "11": "novembro", "12": "dezembro",
}

# Registry of series ingested in Week 1. Each maps to one IBGE aggregate
# table + variable (optionally filtered to one classification category).
SERIES = [
    {
        "name": "IPCA_VARIACAO_MENSAL",
        "label": "IPCA - variação mensal",
        "agregado": 1737,
        "variavel": 63,
        "classificacao": None,
        "period_kind": "monthly",
        "unit": "%",
        "sidra_table": 1737,
    },
    {
        "name": "PIB_VARIACAO_TRIMESTRAL",
        "label": "PIB a preços de mercado - taxa trimestral (mesmo período do ano anterior)",
        "agregado": 5932,
        "variavel": 6561,
        "classificacao": "11255[90707]",
        "period_kind": "quarterly",
        "unit": "%",
        "sidra_table": 5932,
    },
    {
        "name": "PNAD_TAXA_DESOCUPACAO",
        "label": "PNAD Contínua - taxa de desocupação",
        "agregado": 6381,
        "variavel": 4099,
        "classificacao": None,
        "period_kind": "moving_quarter",
        "unit": "%",
        "sidra_table": 6381,
    },
    {
        "name": "PNAD_RENDIMENTO_MEDIO_REAL",
        "label": "PNAD Contínua - rendimento médio mensal real",
        "agregado": 6390,
        "variavel": 5933,
        "classificacao": None,
        "period_kind": "moving_quarter",
        "unit": "R$",
        "unit_position": "prefix",
        "sidra_table": 6390,
    },
    {
        "name": "PNAD_TAXA_INFORMALIDADE",
        "label": "PNAD Contínua - taxa de informalidade",
        "agregado": 6402,
        "variavel": 12466,
        "classificacao": "86[95251]",
        "period_kind": "moving_quarter",
        "unit": "%",
        "sidra_table": 6402,
    },
    {
        "name": "POBREZA_LINHA_INTERNACIONAL",
        "label": "Síntese de Indicadores Sociais - taxa de pobreza pela linha internacional (proporção da população abaixo da linha de pobreza internacional)",
        "agregado": 5817,
        "variavel": 9617,
        "classificacao": None,
        "period_kind": "annual",
        "unit": "%",
        "sidra_table": 5817,
    },
    {
        "name": "POBREZA_LINHA_NACIONAL",
        "label": "Síntese de Indicadores Sociais - taxa de pobreza pela linha nacional (proporção da população abaixo da linha de pobreza nacional)",
        "agregado": 5877,
        "variavel": 9948,
        "classificacao": None,
        "period_kind": "annual",
        "unit": "%",
        "sidra_table": 5877,
    },
    {
        # Fills the pre-2012 gap that PNAD Continua can't (it only starts
        # 2012-03). NOT the same indicator: PME covered only 6 metro areas
        # (never Brazil as a whole) and used a 10+ age base vs PNAD
        # Continua's 14+, so this is kept as an explicitly separate,
        # differently-labeled series rather than spliced into
        # PNAD_TAXA_DESOCUPACAO. See docs/01-decisions.md.
        "name": "PME_TAXA_DESEMPREGO_METROPOLITANA",
        "label": (
            "PME (Pesquisa Mensal de Emprego, descontinuada em 2016) - taxa media anual de "
            "desocupacao, total das 6 regioes metropolitanas (Recife, Salvador, Belo Horizonte, "
            "Rio de Janeiro, Sao Paulo e Porto Alegre), pessoas de 10 anos ou mais"
        ),
        "agregado": 1168,
        "variavel": 2498,
        "classificacao": None,
        "period_kind": "annual",
        "unit": "%",
        "sidra_table": 1168,
        "nivel_territorial": "N110",
        "localidade": "all",
    },
]


def fetch_series(agregado: int, variavel: int, periodos: str = "all",
                  classificacao: str | None = None, nivel_territorial: str = "N1",
                  localidade: str = "1") -> list[tuple[str, str]]:
    """Returns [(period_code, value), ...] sorted as the API returns them
    (chronological). `periodos` follows the IBGE API convention: "all" for
    the full history, or "-N" for the last N periods. `nivel_territorial`/
    `localidade` default to Brazil (N1[1]); the discontinued PME series
    uses N110[all] instead (its own special "all metro areas combined"
    territorial level — see docs/01-decisions.md).
    """
    url = f"{BASE_URL}/{agregado}/periodos/{periodos}/variaveis/{variavel}"
    params = {"localidades": f"{nivel_territorial}[{localidade}]"}
    if classificacao:
        params["classificacao"] = classificacao

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    serie = data[0]["resultados"][0]["series"][0]["serie"]
    return [(period, value) for period, value in serie.items() if value not in _MISSING_VALUES]


def format_period(period_kind: str, period: str) -> str:
    """Turns an IBGE period code into a human-readable Portuguese phrase."""
    if period_kind == "annual":
        return period
    year, tail = period[:4], period[4:]
    if period_kind == "monthly":
        return f"{MONTHS_PT[tail]} de {year}"
    if period_kind == "quarterly":
        return f"{int(tail)}º trimestre de {year}"
    if period_kind == "moving_quarter":
        return f"trimestre móvel encerrado em {MONTHS_PT[tail]} de {year}"
    raise ValueError(f"unknown period_kind: {period_kind}")
