"""BCB lookup agent: deterministic answer for "latest Selic / dollar" questions,
the Banco Central counterpart of the Table Agent. No LLM. Answers ONLY the latest
value; a named period or a mix with an IBGE indicator gets an explicit message
rather than a silent partial answer (the recurring bug in this project).
"""

import re

from src import bcb_client, correlation

_TERMS = {"SELIC_META": ["selic"], "CAMBIO_USD_DIARIO": ["cambio", "dolar"]}
_PERIOD = re.compile(r"\b(19|20)\d\d\b|\b(janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b|\b\d{1,2}/\d{1,2}\b")


def _num(v: float, places: int) -> str:
    return f"{v:.{places}f}".replace(".", ",")


def answer(question: str) -> dict | None:
    q = correlation._norm(question)
    names = [n for n, ts in _TERMS.items() if any(t in q for t in ts)]
    if not names or any(c in q for c in correlation._RELATION_CUES):
        return None  # not a plain lookup: relations belong to the Correlation Agent
    if _PERIOD.search(q):
        return {"answer": "Só consigo informar o valor mais recente da Selic e do câmbio, não um período específico.",
                "citation": "IndicaLens - BCB lookup (escopo atual: valor mais recente)"}
    parts, cites = [], []
    for n in names:
        date, v = bcb_client.fetch_latest(bcb_client.SERIES[n]["code"])
        cite = f"Banco Central, SGS {bcb_client.SERIES[n]['code']}"
        cites.append(cite)
        if n == "SELIC_META":
            parts.append(f"Meta da taxa Selic vigente em {date}: {_num(v, 2)}% a.a. (definida pelo Copom; {cite}).")
        else:
            parts.append(f"Dólar americano (venda, câmbio livre) em {date}: R$ {_num(v, 4)} ({cite}). É a última cotação diária divulgada, não necessariamente a de hoje.")
    if any(t in q for ts in correlation._IBGE_TERMS.values() for t in ts):
        parts.append("Sua pergunta também cita um indicador do IBGE: respondo uma fonte por vez, então pergunte esse indicador separadamente.")
    return {"answer": "\n".join(parts), "citation": "; ".join(cites)}
