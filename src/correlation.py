"""Correlation Agent: does a Banco Central series (Selic, exchange rate) move
together with IPCA? Deterministic, no LLM: the numbers and the wording come
from code, so a claim can't be invented. See docs/06-correlation.md.

Design choices (each one is a way to avoid a misleading answer):
- variations, not levels (a level-vs-level correlation of two trending series
  is spurious): Selic -> change in the monthly rate, exchange rate -> % change,
  IPCA -> its monthly rate (already a variation);
- window from 2000-01 (inflation targeting + floating exchange rate);
- lags 0..12 months (x leads IPCA), and because picking the best of 13 lags
  inflates chance findings, the best lag is reported with a Bonferroni-widened
  interval, next to lag 0;
- never says "causes": Selic also reacts to inflation.
"""

import math
import re
import statistics
import unicodedata
from statistics import NormalDist

from src import bcb_client
from src.ibge_client import fetch_series as ibge_fetch

START = "200001"
MAX_LAG = 12
IPCA = {"agregado": 1737, "variavel": 63}

_BCB_TERMS = {
    "SELIC_MENSAL": ["selic", "juros"],
    "CAMBIO_USD_MEDIA_MENSAL": ["cambio", "dolar"],
}
_IBGE_TERMS = {"ipca": ["ipca", "inflacao"], "outro": ["pib", "desocupacao", "desemprego", "rendimento", "renda", "salario", "informal", "pobreza"]}
_POINT_VALUE = re.compile(r"\d{2,4}|recente|ultimo|hoje|atual|vigente|agora")
_RELATION_CUES = ["junto", "se move", "move com", "relacao", "relaciona", "correlac", "afeta", "influenc", "impact",
                  "antecede", "depende", "associa", "acompanha", "reage", "sensivel", "anda com"]


def _norm(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def add_months(period: str, k: int) -> str:
    n = int(period[:4]) * 12 + int(period[4:]) - 1 + k
    return f"{n // 12}{n % 12 + 1:02d}"


def _diff(s: dict[str, float]) -> dict[str, float]:
    return {p: v - s[add_months(p, -1)] for p, v in s.items() if add_months(p, -1) in s}


def _pct_change(s: dict[str, float]) -> dict[str, float]:
    return {p: 100 * (v / s[add_months(p, -1)] - 1) for p, v in s.items() if add_months(p, -1) in s}


def _ci(r: float, n: int, alpha: float) -> tuple[float, float]:
    z, se, crit = math.atanh(r), 1 / math.sqrt(n - 3), NormalDist().inv_cdf(1 - alpha / 2)
    return math.tanh(z - crit * se), math.tanh(z + crit * se)


def analyze(x: dict[str, float], y: dict[str, float], start: str = START, max_lag: int = MAX_LAG) -> dict:
    """x leads y by `lag` months. Returns per-lag r/n/95% CI and the best lag
    (largest |r|) with a Bonferroni-adjusted CI over the lags tested."""
    lags = []
    for k in range(max_lag + 1):
        pairs = [(x[add_months(t, -k)], y[t]) for t in y if t >= start and add_months(t, -k) in x]
        if len(pairs) < 30:
            continue
        try:
            r = statistics.correlation([a for a, _ in pairs], [b for _, b in pairs])
        except statistics.StatisticsError:  # a constant input has no correlation
            continue
        lo, hi = _ci(r, len(pairs), 0.05)
        lags.append({"lag": k, "n": len(pairs), "r": r, "lo": lo, "hi": hi})
    if not lags:
        raise ValueError("not enough overlapping months")
    best = max(lags, key=lambda d: abs(d["r"]))
    best = {**best}
    best["lo_adj"], best["hi_adj"] = _ci(best["r"], best["n"], 0.05 / len(lags))
    best["distinguishable"] = not (best["lo_adj"] <= 0 <= best["hi_adj"])
    return {"lags": lags, "best": best, "lag0": lags[0] if lags[0]["lag"] == 0 else None}


def _strength(r: float) -> str:
    a = abs(r)
    return "praticamente nula" if a < 0.1 else "fraca" if a < 0.3 else "moderada" if a < 0.5 else "forte"


def _num(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def describe(bcb_name: str, res: dict) -> str:
    s = bcb_client.SERIES[bcb_name]
    what = {"SELIC_MENSAL": "variação mensal da Selic", "CAMBIO_USD_MEDIA_MENSAL": "variação percentual mensal do dólar"}[bcb_name]
    b, l0 = res["best"], res["lag0"]
    lines = [f"Correlação entre a {what} e o IPCA mensal (jan/2000 em diante):"]
    if l0:
        lines.append(f"- Mesmo mês: r = {_num(l0['r'])} (IC 95%: {_num(l0['lo'])} a {_num(l0['hi'])}; n = {l0['n']} meses).")
    lines.append(
        f"- Maior correlação entre as defasagens 0 a {res['lags'][-1]['lag']} meses: {b['lag']} mês(es) de defasagem, "
        f"r = {_num(b['r'])} ({_strength(b['r'])}; IC ajustado por múltiplas defasagens: {_num(b['lo_adj'])} a {_num(b['hi_adj'])}; n = {b['n']})."
    )
    lines.append(
        "- Diferente de zero mesmo depois de ajustar pelas defasagens testadas." if b["distinguishable"]
        else "- Não é distinguível de zero depois de ajustar pelas defasagens testadas: não há evidência de relação linear."
    )
    reverse = {"SELIC_MENSAL": "a Selic também reage à inflação", "CAMBIO_USD_MEDIA_MENSAL": "o câmbio também reage às expectativas de inflação e juros"}[bcb_name]
    lines.append(
        f"Correlação não prova causa: {reverse}, e ambos respondem a outros fatores. "
        "Meses vizinhos não são independentes, então os intervalos tendem a ser otimistas."
    )
    lines.append(f"Fontes: IBGE/SIDRA tabela 1737 (IPCA); Banco Central, SGS {s['code']} ({s['label']}).")
    return "\n".join(lines)


def detect(question: str) -> tuple[str | None, str | None] | None:
    """(bcb_series_name, ibge_kind) if the question asks how a BCB series
    relates to an IBGE indicator, else None (not ours: let other agents handle it)."""
    q = _norm(question)
    bcb = next((n for n, ts in _BCB_TERMS.items() if any(t in q for t in ts)), None)
    ibge = next((k for k, ts in _IBGE_TERMS.items() if any(t in q for t in ts)), None)
    if not (bcb and ibge):
        return None
    # a series of each side in one question is a relation unless it asks for a point value
    # (a period, "mais recente", "hoje"...); explicit relation cues always count
    if any(c in q for c in _RELATION_CUES) or not _POINT_VALUE.search(q):
        return bcb, ibge
    return None


def answer(question: str) -> dict | None:
    hit = detect(question)
    if hit is None:
        return None
    bcb_name, ibge_kind = hit
    if ibge_kind != "ipca":
        return {"answer": "Por enquanto só sei relacionar a Selic e o câmbio com o IPCA (inflação mensal). "
                          "Não vou estimar a relação com esse outro indicador.",
                "citation": "IndicaLens - Correlation Agent (escopo atual: IPCA)"}
    ipca = {p: float(v) for p, v in ibge_fetch(IPCA["agregado"], IPCA["variavel"], "all")}
    raw = dict(bcb_client.fetch_series(bcb_client.SERIES[bcb_name]["code"]))
    x = _diff(raw) if bcb_name == "SELIC_MENSAL" else _pct_change(raw)
    res = analyze(x, ipca)
    code = bcb_client.SERIES[bcb_name]["code"]
    return {"answer": describe(bcb_name, res),
            "citation": f"IBGE/SIDRA tabela 1737 (IPCA); Banco Central SGS {code}",
            "result": res}
