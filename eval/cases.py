"""Eval cases. Two tiers:

- offline: Table Agent only (deterministic, needs just the public IBGE API,
  no secrets), so it can gate every PR in CI. Expected values are NOT
  hard-coded: they come from an independent fetch of the same series at eval
  time, so a data revision by the IBGE doesn't fake a regression. What the
  cases really test is the parsing/routing logic, where every past bug lived.
- full: the whole qa_agent graph (needs Supabase + Groq), plus planted-error
  cases that measure the Critic itself. The hand-written expectations in
  FULL_* are anchors to review and grow, not ground truth from the IBGE.
"""

from src.ibge_client import MONTHS_PT, SERIES, fetch_series, format_period

_BY_NAME = {s["name"]: s for s in SERIES}

# series -> question templates ({when} is filled per period kind)
_TEMPLATES = {
    "IPCA_VARIACAO_MENSAL": ["Qual foi o IPCA em {when}?", "Qual a inflacao (IPCA) de {when}?"],
    "PIB_VARIACAO_TRIMESTRAL": ["Qual foi o PIB no {when}?"],
    "PNAD_TAXA_DESOCUPACAO": ["Qual foi a taxa de desocupacao em {when}?"],
    "PNAD_RENDIMENTO_MEDIO_REAL": ["Qual foi o rendimento medio em {when}?"],
    "PNAD_TAXA_INFORMALIDADE": ["Qual foi a taxa de informalidade em {when}?"],
    "POBREZA_LINHA_INTERNACIONAL": ["Qual foi a taxa de pobreza internacional em {when}?"],
    "POBREZA_LINHA_NACIONAL": ["Qual foi a taxa de pobreza nacional em {when}?"],
    "PME_TAXA_DESEMPREGO_METROPOLITANA": ["Qual foi a taxa de desemprego da PME em {when}?"],
}
_LATEST = {
    "IPCA_VARIACAO_MENSAL": "Qual foi o IPCA mais recente?",
    "PIB_VARIACAO_TRIMESTRAL": "Qual foi o PIB mais recente?",
    "PNAD_TAXA_DESOCUPACAO": "Qual foi a taxa de desocupacao mais recente?",
    "PNAD_RENDIMENTO_MEDIO_REAL": "Qual foi o rendimento medio mais recente?",
    "PNAD_TAXA_INFORMALIDADE": "Qual foi a taxa de informalidade mais recente?",
    "POBREZA_LINHA_INTERNACIONAL": "Qual foi a taxa de pobreza internacional mais recente?",
    "POBREZA_LINHA_NACIONAL": "Qual foi a taxa de pobreza nacional mais recente?",
    "PME_TAXA_DESEMPREGO_METROPOLITANA": "Qual foi a taxa de desemprego da PME mais recente?",
}
_SAMPLES = {  # how many evenly spread periods to test per series
    "IPCA_VARIACAO_MENSAL": 12, "PIB_VARIACAO_TRIMESTRAL": 8, "PNAD_TAXA_DESOCUPACAO": 8,
    "PNAD_RENDIMENTO_MEDIO_REAL": 6, "PNAD_TAXA_INFORMALIDADE": 6, "POBREZA_LINHA_INTERNACIONAL": 4,
    "POBREZA_LINHA_NACIONAL": 4, "PME_TAXA_DESEMPREGO_METROPOLITANA": 4,
}


def _phrase(kind: str, period: str) -> str:
    """How a person writes the period, not how format_period writes it."""
    if kind == "moving_quarter":
        return f"{MONTHS_PT[period[4:]]} de {period[:4]}"
    return format_period(kind, period)


def _fmt(series: dict, value: str) -> str:
    return f"{series['unit']} {value}" if series.get("unit_position") == "prefix" else f"{value}{series['unit']}"


def lookup_cases() -> list[dict]:
    """[{question, expected_value, expected_period, series}] from live data."""
    cases = []
    for name, series in _BY_NAME.items():
        points = fetch_series(
            series["agregado"], series["variavel"], "all", series["classificacao"],
            nivel_territorial=series.get("nivel_territorial", "N1"), localidade=series.get("localidade", "1"),
        )
        cases.append({"question": _LATEST[name], "expected_value": _fmt(series, points[-1][1]),
                      "expected_period": format_period(series["period_kind"], points[-1][0]).lower(),
                      "series": name, "kind": "latest"})
        n = _SAMPLES[name]
        step = max(1, len(points) // n)
        for i, (period, value) in enumerate(points[step // 2::step][:n]):
            template = _TEMPLATES[name][i % len(_TEMPLATES[name])]
            cases.append({
                "question": template.format(when=_phrase(series["period_kind"], period)),
                "expected_value": _fmt(series, value),
                "expected_period": format_period(series["period_kind"], period).lower(),
                "series": name, "kind": "period",
            })
    return cases


# Questions the Table Agent must decline (return None) instead of guessing.
REFUSAL_CASES = [
    # relational / non-IBGE-series questions: must not return the latest value of the one series named
    "A inflacao se move junto com a Selic?", "O cambio afeta o IPCA?", "Qual a relacao entre juros e desocupacao?",
    "O dolar influencia a inflacao?", "A taxa de desocupacao depende da Selic?",
    # a bare year is ambiguous for monthly / quarterly / moving-quarter series
    "Qual foi o IPCA em 2010?", "Qual foi o PIB em 1999?", "Qual foi a taxa de desocupacao em 2024?",
    "Qual foi o rendimento medio em 2020?", "Qual foi a taxa de informalidade em 2022?",
    # before the series starts
    "Qual foi a taxa de desocupacao em janeiro de 2010?", "Qual foi a taxa de desocupacao em 1995?",
    "Qual foi o rendimento medio em marco de 2005?", "Qual foi a taxa de informalidade em maio de 2000?",
    "Qual foi a taxa de pobreza nacional em 1990?", "Qual foi a taxa de desemprego da PME em 1999?",
    # two periods in one question
    "Qual foi o IPCA em janeiro de 1998 e de 1999?", "Qual foi o PIB no 1 trimestre de 2019 e no 2 trimestre de 2020?",
    "Qual foi a taxa de pobreza internacional em 2015 e 2016?",
    # out of domain
    "Qual a capital da Franca?", "Qual o preco do Bitcoin hoje?", "Qual o resultado do jogo do Brasil ontem?",
    # analytical, not a value lookup
    "Como a taxa de desocupacao varia por nivel de instrucao?", "Por que a taxa de desocupacao caiu no ultimo trimestre?",
    "Quais fatores explicam a alta do IPCA em 2021?",
]

# --- full tier (hand-written; review and extend) ------------------------------

FULL_ROUTE = [  # must be answered by the deterministic path (score 1.0, no LLM)
    "Qual foi o IPCA mais recente?", "Como esta a taxa de desemprego no Brasil?",
    "Qual foi o rendimento medio mais recente?", "Qual foi a taxa de pobreza internacional mais recente?",
    "Qual foi a taxa de desocupacao de maio de 2025?", "Qual foi a taxa de desemprego da PME em 2010?",
    "Qual a taxa selic mais recente?", "Qual o cambio mais recente para dolar?",
]
FULL_REVISION = [  # (question, must appear in the answer)
    ("O PIB do 2 trimestre de 2022 foi revisado?", "3.5%"),
    ("O PIB do 2 trimestre de 2020 sofreu revisao?", "-11.4%"),
    ("O PIB do 3 trimestre de 2023 foi revisado?", "primeira divulgacao"),
    ("O PIB do 1 trimestre de 2026 foi revisado?", "sem revisao"),
]
FULL_DECLINE = [  # the graph must admit lack of data, not invent
    "Por que a taxa de desocupacao caiu no ultimo trimestre?", "Quais fatores explicam a alta do IPCA em 2021?",
    "Como o rendimento medio se distribui por regiao no Brasil?", "Qual a capital da Franca?",
    "Qual o preco do Bitcoin hoje?", "Qual foi o PIB em 1999?", "Qual foi a taxa de desocupacao em 1995?",
    "Qual foi a taxa de desemprego no Brasil em 1990?",
]
FULL_MULTI = [("Qual foi o IPCA em janeiro de 1998 e de 1999?", ["1998", "1999"]),
              ("Qual foi a taxa de pobreza internacional em 2015 e 2016?", ["2015", "2016"])]
FULL_PROSE = [  # (question, at least one of these fragments must appear; snapshot-style anchors)
    ("Como a taxa de desocupacao varia por nivel de instrucao no Brasil?", ["medio incompleto", "superior"]),
    ("O que os comentarios do IBGE dizem sobre a taxa de desocupacao no ultimo trimestre?", ["5,4", "5.4", "reducao", "queda"]),
    ("O que o IBGE informa sobre o rendimento medio real habitual?", ["rendimento", "3.7", "3,7"]),
    # values below are the ones printed in the PNAD 2T/2026 commentary chunks
    ("Qual percentual da populacao total era populacao em idade de trabalhar no 2 trimestre de 2026?", ["82,1", "82.1"]),
    ("Qual a participacao de homens entre as pessoas ocupadas?", ["56,6", "56.6"]),
    ("Qual foi a taxa de participacao na forca de trabalho no 2 trimestre de 2026?", ["62,1", "62.1"]),
    ("Qual percentual das pessoas em idade de trabalhar estava fora da forca de trabalho?", ["37,8", "37.8"]),
    ("Quantas pessoas estavam fora da forca de trabalho no 2 trimestre de 2026?", ["66,5", "66.5"]),
]


def critic_cases() -> list[dict]:
    """Planted-error cases that measure the Critic itself (evaluating the
    evaluator). Built from real IPCA points so the contexts are realistic.
    """
    series = _BY_NAME["IPCA_VARIACAO_MENSAL"]
    points = fetch_series(series["agregado"], series["variavel"], "all")
    picks = points[10:: max(1, len(points) // 15)][:15]
    cases = []
    for i, (period, value) in enumerate(picks):
        when = format_period("monthly", period)
        v = float(value)
        ctx = [{"content": f"Segundo o IBGE (SIDRA, tabela 1737), IPCA - variação mensal em {when} foi de {value}%."}]
        pt = lambda x: f"{x:.2f}".replace(".", ",")
        good = f"O IPCA em {when} foi de {pt(v)} % [1]."
        bad = f"O IPCA em {when} foi de {pt(v + 0.4)} % [1]."
        invented = good + " Esse foi o maior valor da serie historica, por causa da crise internacional."
        cases.append({"ctx": ctx, "answer": good, "expect": ["supported"]})
        if i < 10:
            cases.append({"ctx": ctx, "answer": bad, "expect": ["flagged"]})
        if i < 5:
            cases.append({"ctx": ctx, "answer": invented, "expect": ["supported", "flagged"]})
    return cases


# --- Correlation Agent / BCB lookup (semana 9-10) -----------------------------------

# Golden facts about the BCB series, hard-coded from public knowledge (NOT computed by the code
# under test): (question, substrings that must all appear in the answer).
BCB_GOLDEN = [
    ("Qual a taxa selic em 31/12/2022?", ["13,75% a.a."]),
    ("Qual a taxa selic em 31/12/2021?", ["9,25% a.a."]),
    ("Qual a taxa selic em 31/12/2019?", ["4,50% a.a."]),
    ("qual a taxa selic de novembro de 2022?", ["13,75% a.a.", "1,02%"]),
    ("selic de 2022", ["terminou o ano em 13,75%", "entre 9,25% e 13,75%"]),
    ("selic em 2021 e 2022", ["terminou o ano em 9,25%", "terminou o ano em 13,75%"]),
    ("qual o cambio em 30/12/2022 para dolar?", ["R$ 5,2177"]),
    ("dolar em 31/12/2021", ["R$ 5,5805"]),
    ("dolar em 30/12/2020", ["R$ 5,1967"]),
    ("dolar em 05/03/2022", ["R$ 5,0758", "04/03/2022"]),  # a Saturday: last quotation before it, and says so
    ("qual o cambio em dezembro de 2022 para dolar?", ["R$ 5,2177 em 30/12/2022"]),
]
# Inputs that must NOT produce a number for a different period (message substring expected).
BCB_REFUSALS = [
    ("selic em março", "informe o ano"),
    ("Qual a selic em 30/02/2022?", "não existe"),
    ("selic em dezembro de 2099", "futura"),
    ("dolar em 2099", "futura"),
    ("selic em 1º de janeiro de 1970", "sem dados"),
]

# Recognition of correlation questions in varied phrasing. DEV was used to tune the cue list;
# TEST was written at the same time and is only scored, never tuned on. (question, is_correlation)
CORR_ROUTE_DEV = [
    ("A inflacao se move junto com a Selic?", True), ("O cambio afeta o IPCA?", True),
    ("Existe relacao entre juros e inflacao?", True), ("O dolar influencia a inflacao?", True),
    ("A Selic tem correlacao com o IPCA?", True), ("O IPCA depende do cambio?", True),
    ("A inflacao acompanha o dolar?", True), ("Qual o impacto do cambio no IPCA?", True),
    ("A taxa de juros antecede a inflacao?", True), ("A inflacao reage a Selic?", True),
    ("Qual o valor da Selic hoje?", False), ("Qual foi o IPCA em janeiro de 1998?", False),
    ("Por que a inflacao subiu em 2021?", False), ("Qual foi a taxa de desocupacao mais recente?", False),
    ("Qual o cambio mais recente para dolar?", False), ("O que os comentarios do IBGE dizem sobre renda?", False),
]
CORR_ROUTE_DEV += [
    ("O dolar pressiona a inflacao?", True), ("Qual o efeito da Selic sobre o IPCA?", True),
    ("Juros altos derrubam a inflacao?", True), ("Ha relacao entre cambio e IPCA?", True),
    ("A Selic e o IPCA andam juntos?", True), ("A inflacao esta associada ao cambio?", True),
    ("Como a Selic se relaciona com a inflacao?", True), ("O IPCA responde ao dolar?", True),
    ("Quando o dolar sobe, a inflacao sobe?", True), ("A inflacao segue a taxa de juros?", True),
    ("Qual a Selic vigente?", False), ("Qual foi o PIB no ultimo trimestre?", False),
    ("Qual a taxa de informalidade mais recente?", False), ("O PIB de 2020 foi revisado?", False),
    ("Qual foi o dolar em 30/12/2022?", False), ("Como a renda se distribui por regiao?", False),
]
CORR_ROUTE_TEST = [  # written BEFORE the rule change below and never used for tuning (v2; v1 became DEV_2)
    ("O cambio impulsiona a inflacao?", True), ("A alta da Selic freia o IPCA?", True),
    ("Existe ligacao entre o dolar e a inflacao?", True), ("O IPCA varia conforme os juros?", True),
    ("Quando a Selic aumenta, o que acontece com a inflacao?", True), ("O dolar mais caro encarece o IPCA?", True),
    ("Juros e inflacao caminham na mesma direcao?", True), ("A inflacao muda quando o cambio muda?", True),
    ("Qual o IPCA e a Selic mais recentes?", False), ("Qual foi o IPCA e o dolar em 2022?", False),
    ("Me diga o dolar de hoje e a inflacao do ultimo mes", False), ("Qual a meta da Selic atual?", False),
    ("O PIB cresceu mais que a inflacao em 2021?", False), ("Qual foi a variacao do dolar em 2022?", False),
]
