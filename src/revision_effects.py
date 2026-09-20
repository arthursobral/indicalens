"""Do GDP revisions move the Selic target or the exchange rate afterwards? Deterministic, no LLM.

Event = one IBGE release (booklet). News at release t = sum, over the quarters the booklet shares
with the previous one (all but the newest), of (value in this booklet - value in the previous one),
in percentage points. Every release covers the same ages (1 to 4 quarters), so the measure does not
drift with how old a quarter is (the confound of "current value - first release").

Outcomes after the release date: change in the Selic target (SGS 432, p.p.) and % change in the
dollar (SGS 1). Placebo: the same outcomes in the window BEFORE the release; an "effect" that also
appears before the news is trend, not response. With ~25 events only a large association is
detectable; the answer says so. See docs/06-correlation.md section 9.
"""

import random
import statistics
from datetime import date, timedelta

from src import bcb_client
from src.ibge_client import MONTHS_PT
from src.revisions import load_vintages, same_age_revisions

H = 90  # days
SELIC, USD = 432, 1


def events(vintages: list[dict]) -> list[dict]:
    out = []
    for prev, cur in zip(vintages, vintages[1:]):
        shared = [q for q in cur["table"] if q in prev["table"] and q != cur["period"]]
        news = round(sum(cur["table"][q] - prev["table"][q] for q in shared), 1)
        out.append({"period": cur["period"], "date": date.fromisoformat(cur["released"]), "news": news, "n_quarters": len(shared)})
    return out


def _at(series: list[tuple[date, float]], d: date) -> float | None:
    """Last observation on or before d."""
    best = None
    for day, v in series:
        if day > d:
            break
        best = v
    return best


def load_market(evs: list[dict]) -> tuple[list, list]:
    start, end = min(e["date"] for e in evs) - timedelta(days=H + 10), max(e["date"] for e in evs) + timedelta(days=H)
    return bcb_client.fetch_range(SELIC, start, end), bcb_client.fetch_range(USD, start, end)


def outcomes(evs: list[dict], selic: list, usd: list, today: date | None = None) -> dict[str, list[tuple[float, float]]]:
    """{outcome_name: [(news, outcome), ...]} keeping only events whose window is complete."""
    today = today or date.today()
    spec = {
        "selic_depois": lambda t: (_at(selic, t + timedelta(days=H)), _at(selic, t), lambda a, b: a - b),
        "selic_antes": lambda t: (_at(selic, t), _at(selic, t - timedelta(days=H)), lambda a, b: a - b),
        "dolar_depois": lambda t: (_at(usd, t + timedelta(days=H)), _at(usd, t), lambda a, b: 100 * (a / b - 1)),
        "dolar_antes": lambda t: (_at(usd, t), _at(usd, t - timedelta(days=H)), lambda a, b: 100 * (a / b - 1)),
    }
    res = {k: [] for k in spec}
    for e in evs:
        for k, f in spec.items():
            a, b, fn = f(e["date"])
            complete = e["date"] + timedelta(days=H) <= today if k.endswith("depois") else True
            if a is not None and b is not None and complete:
                res[k].append((e["news"], fn(a, b)))
    return res


def stats(pairs: list[tuple[float, float]], perms: int = 10000, seed: int = 0) -> dict:
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(pairs)
    r = statistics.correlation(xs, ys)
    rho = statistics.correlation(xs, ys, method="ranked")
    rnd, ys2, hits = random.Random(seed), list(ys), 0
    for _ in range(perms):
        rnd.shuffle(ys2)
        hits += abs(statistics.correlation(xs, ys2)) >= abs(r) - 1e-12
    return {"n": n, "r": r, "rho": rho, "p_perm": (hits + 1) / (perms + 1)}


def analyze(evs: list[dict], selic: list, usd: list, today: date | None = None) -> dict:
    return {k: stats(v) for k, v in outcomes(evs, selic, usd, today).items()}


def _num(v: float, places: int = 2) -> str:
    return f"{v:.{places}f}".replace(".", ",")


def describe(res: dict, vintages: list[dict]) -> str:
    k1, k4 = same_age_revisions(vintages, 1), same_age_revisions(vintages, 4)
    evs = events(vintages)
    with_news = [e for e in evs if e["news"] != 0]
    months = sorted({e["date"].month for e in with_news})
    up4 = sum(r["revision"] > 0 for r in k4)
    nz4 = sum(r["revision"] != 0 for r in k4)
    lines = [
        "Revisões do PIB (comparando o mesmo trimestre com a mesma idade, para não confundir revisão com tempo decorrido):",
        f"- Na divulgação seguinte, {sum(r['revision'] != 0 for r in k1)} de {len(k1)} trimestres foram revisados"
        + (" (todos segundos trimestres)." if {r["period"][4:] for r in k1 if r["revision"] != 0} == {"02"} else "."),
        f"- Um ano depois (4 divulgações), {nz4} de {len(k4)} foram revisados, {up4} para cima (média {_num(statistics.mean(r['revision'] for r in k4), 2)} p.p.).",
        "",
        f"Só {len(with_news)} de {len(evs)} divulgações trazem alguma revisão de trimestres anteriores"
        + (f", todas em {MONTHS_PT[f'{months[0]:02d}']}." if len(months) == 1 else f" (meses: {', '.join(MONTHS_PT[f'{m:02d}'] for m in months)})."),
        f"Associação entre a \"notícia de revisão\" de cada divulgação e o que aconteceu em {H} dias (n = {res['selic_depois']['n']} divulgações; "
        "r de Pearson, ρ de Spearman, p por permutação):",
    ]
    for name, label in (("selic_depois", "Meta Selic depois"), ("dolar_depois", "Dólar depois"), ("selic_antes", "Meta Selic antes (controle)"), ("dolar_antes", "Dólar antes (controle)")):
        s = res[name]
        lines.append(f"- {label}: r = {_num(s['r'])}, ρ = {_num(s['rho'])}, p = {_num(s['p_perm'])} (n = {s['n']}).")
    hit = [k for k in ("selic_depois", "dolar_depois") if res[k]["p_perm"] < 0.05]
    lines.append(
        "Resultado: " + ("há associação com " + " e ".join(hit) + ", mas confira os controles: se o mesmo aparece antes da divulgação, é tendência e não resposta."
                        if hit else "não há evidência de que as revisões do PIB precedam movimentos da Selic ou do dólar.")
    )
    lines.append(
        f"Limites: com {res['selic_depois']['n']} divulgações, e só {len(with_news)} com revisão, apenas uma associação grande (|r| acima de ~0,4) seria detectável, "
        "então \"sem evidência\" não é \"sem efeito\"; "
        "a Selic e o dólar reagem a muitas outras coisas ao mesmo tempo; as revisões vêm em blocos anuais."
    )
    lines.append("Fontes: IBGE, Contas Nacionais Trimestrais (cadernos); Banco Central, SGS 432 (meta Selic) e SGS 1 (dólar).")
    return "\n".join(lines)


_BCB = ("selic", "juros", "cambio", "dolar")


def matches(q: str) -> bool:
    return "revis" in q and "pib" in q and any(t in q for t in _BCB)


def answer(question: str) -> dict | None:
    from src.correlation import _norm

    if not matches(_norm(question)):
        return None
    vint = load_vintages()
    evs = events(vint)
    selic, usd = load_market(evs)
    res = analyze(evs, selic, usd)
    return {"answer": describe(res, vint), "citation": "IBGE Contas Nacionais Trimestrais; Banco Central SGS 432 e 1", "result": res}
