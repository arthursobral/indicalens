"""BCB lookup agent: deterministic answers about the Selic target and the dollar,
the Banco Central counterpart of the Table Agent. No LLM.

Handles: latest value, an exact date, a month, a year, or several of those. Each
granularity has ONE stated rule (a bare year is ambiguous for a series that moves
inside the year, so the answer gives year-end and the range and says so). A period
we can't resolve (month without year, future, no data) gets an explicit message,
never a silent answer for a different period.
"""

import calendar
import re
from datetime import date, timedelta

from src import bcb_client, correlation

_TERMS = {"SELIC_META": ["selic"], "CAMBIO_USD_DIARIO": ["cambio", "dolar"]}
_MONTHS = {m: i for i, m in enumerate(
    ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"], 1)}
_MONTH_RE = "|".join(_MONTHS)
_DATE_NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d\d)\b")
_DATE_TXT = re.compile(rf"\b(\d{{1,2}})[ºo]? de ({_MONTH_RE}) de ((?:19|20)\d\d)\b")
_MONTH_YEAR = re.compile(rf"\b({_MONTH_RE})(?: de| /)? ((?:19|20)\d\d)\b")
_YEAR = re.compile(r"\b((?:19|20)\d\d)\b")
_MAX_PERIODS = 6


def _num(v: float, places: int) -> str:
    return f"{v:.{places}f}".replace(".", ",")


def _br(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def parse_periods(q: str) -> list[tuple] | str:
    """[('date', d) | ('month', y, m) | ('year', y), ...] or an error message. `q` is normalized."""
    found = []
    for rx, build in ((_DATE_NUM, lambda g: ("date", g[2], g[1], g[0])), (_DATE_TXT, lambda g: ("date", g[2], _MONTHS[g[1]], g[0]))):
        for m in rx.finditer(q):
            _, y, mo, d = build(m.groups())
            try:
                found.append(("date", date(int(y), int(mo), int(d))))
            except ValueError:
                return f"A data {m.group(0)} não existe."
        q = rx.sub(" ", q)
    for m in _MONTH_YEAR.finditer(q):
        found.append(("month", int(m.group(2)), _MONTHS[m.group(1)]))
    q = _MONTH_YEAR.sub(" ", q)
    if re.search(rf"\b({_MONTH_RE})\b", q):
        return "Você citou um mês sem o ano. Informe o ano (por exemplo, \"março de 2022\")."
    found += [("year", int(y)) for y in _YEAR.findall(q)]
    return found


def _window(p: tuple) -> tuple[date, date]:
    if p[0] == "date":
        return p[1] - timedelta(days=10), p[1]  # 10 days back covers weekends/holidays
    if p[0] == "month":
        return date(p[1], p[2], 1), date(p[1], p[2], calendar.monthrange(p[1], p[2])[1])
    return date(p[1], 1, 1), date(p[1], 12, 31)


def _label(p: tuple) -> str:
    return _br(p[1]) if p[0] == "date" else f"{list(_MONTHS)[p[2] - 1]} de {p[1]}" if p[0] == "month" else str(p[1])


def _line(name: str, p: tuple, today: date) -> str:
    start, end = _window(p)
    label = _label(p)
    if (p[1] if p[0] == "date" else date(p[1], p[2] if p[0] == "month" else 1, 1)) > today:
        return f"{label}: ainda não há dados (data futura)."
    end = min(end, today)
    s = bcb_client.SERIES[name]
    cite = f"Banco Central, SGS {s['code']}"
    try:
        rows = bcb_client.fetch_range(s["code"], start, end)
    except bcb_client.requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise  # a real outage must not read as "no data"
        rows = []  # SGS answers 404 for a window before the series exists
    if not rows:
        return f"{label}: sem dados do Banco Central para esse período."
    last_d, last_v = rows[-1]
    vals = [v for _, v in rows]
    if name == "SELIC_META":
        if p[0] == "date":
            note = "" if last_d == p[1] else f" (último dado disponível até essa data: {_br(last_d)})"
            return f"Meta Selic em {label}: {_num(last_v, 2)}% a.a.{note} ({cite})."
        if p[0] == "month":
            acc = bcb_client.fetch_range(4390, start, end)
            extra = f"; Selic acumulada no mês: {_num(acc[-1][1], 2)}% (SGS 4390)" if acc else ""
            return f"Meta Selic ao fim de {label}: {_num(last_v, 2)}% a.a. (em {_br(last_d)}){extra} ({cite})."
        return (f"Meta Selic em {label}: terminou o ano em {_num(last_v, 2)}% a.a. (em {_br(last_d)}); "
                f"variou entre {_num(min(vals), 2)}% e {_num(max(vals), 2)}% a.a. ao longo do ano ({cite}).")
    if p[0] == "date":
        note = "" if last_d == p[1] else f" (última cotação divulgada até essa data: {_br(last_d)})"
        return f"Dólar americano (venda, câmbio livre) em {label}: R$ {_num(last_v, 4)}{note} ({cite})."
    if p[0] == "month":
        return (f"Dólar americano (venda, câmbio livre) em {label}: média R$ {_num(sum(vals) / len(vals), 4)} "
                f"(cotações diárias); última cotação do mês R$ {_num(last_v, 4)} em {_br(last_d)} ({cite}).")
    return (f"Dólar americano (venda, câmbio livre) em {label}: última cotação R$ {_num(last_v, 4)} em {_br(last_d)}; "
            f"variou entre R$ {_num(min(vals), 4)} e R$ {_num(max(vals), 4)} no ano ({cite}).")


def answer(question: str) -> dict | None:
    q = correlation._norm(question)
    names = [n for n, ts in _TERMS.items() if any(t in q for t in ts)]
    if not names or any(c in q for c in correlation._RELATION_CUES):
        return None  # not a plain lookup: relations belong to the Correlation Agent
    periods = parse_periods(q)
    if isinstance(periods, str):
        return {"answer": periods, "citation": "IndicaLens - BCB lookup"}
    if len(periods) > _MAX_PERIODS:
        return {"answer": f"Informe no máximo {_MAX_PERIODS} períodos por pergunta.", "citation": "IndicaLens - BCB lookup"}
    today = date.today()
    parts, cites = [], [f"Banco Central, SGS {bcb_client.SERIES[n]['code']}" for n in names]
    for n in names:
        if periods:
            parts += [_line(n, p, today) for p in periods]
        else:
            d, v = bcb_client.fetch_latest(bcb_client.SERIES[n]["code"])
            cite = f"Banco Central, SGS {bcb_client.SERIES[n]['code']}"
            if n == "SELIC_META":
                parts.append(f"Meta da taxa Selic vigente em {d}: {_num(v, 2)}% a.a. (definida pelo Copom; {cite}).")
            else:
                parts.append(f"Dólar americano (venda, câmbio livre) em {d}: R$ {_num(v, 4)} ({cite}). É a última cotação diária divulgada, não necessariamente a de hoje.")
    if any(t in q for ts in correlation._IBGE_TERMS.values() for t in ts):
        parts.append("Sua pergunta também cita um indicador do IBGE: respondo uma fonte por vez, então pergunte esse indicador separadamente.")
    return {"answer": "\n".join(parts), "citation": "; ".join(cites)}
