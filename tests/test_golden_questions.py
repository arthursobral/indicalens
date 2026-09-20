"""Golden set: ~50 real questions this project must keep answering
correctly, covering the Table Agent's deterministic lookups, its
deliberate refusals (ambiguous/out-of-range/multi-period/analytical
questions that must defer to RAG instead of guessing), and the full
qa_agent pipeline's routing + RAG fallback behavior.

Hits live infra (IBGE API, Supabase, Groq) - NOT wired into CI: no
GROQ_API_KEY there, and this is slow (dozens of LLM calls, see
test_indicator_analyst.py for the same reasoning). Run locally:
    python tests/test_golden_questions.py

Known flake, not a bug: the QA Agent section fires ~10+ LLM calls back to
back, which can trip Groq's free-tier rate limit on the last one or two
(HTTPError 429). If a case fails with that specific exception, re-run just
that question (e.g. via src.qa_agent.ask) after a short pause instead of
treating it as a regression.
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.table_agent import answer as table_answer
from src.qa_agent import ask as qa_ask


def _norm(s: str) -> str:
    """Lowercase + strip accents, so checks survive PT-BR spelling variance."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def contains(*subs: str):
    """Checker: answer text must contain ALL given substrings (accent-insensitive)."""
    def check(result) -> tuple[bool, str]:
        if result is None:
            return False, "esperava resposta, veio None"
        text = _norm(result["answer"])
        missing = [s for s in subs if _norm(s) not in text]
        if missing:
            return False, f"faltando {missing!r} em: {result['answer']!r}"
        return True, result["answer"]
    return check


def is_none():
    """Checker: table_agent must decline (return None)."""
    def check(result) -> tuple[bool, str]:
        if result is not None:
            return False, f"esperava None (recusa), veio: {result!r}"
        return True, "None (recusou, como esperado)"
    return check


_DECLINE_MARKERS = ["nao ha dados", "nao tenho dados", "nao ha dado"]


def has_decline_phrase():
    """Checker for qa_agent.ask(): final answer must admit lack of data,
    not fabricate one (accent/case-insensitive substring match)."""
    def check(result) -> tuple[bool, str]:
        text = _norm(result["answer"])
        if not any(m in text for m in _DECLINE_MARKERS):
            return False, f"esperava recusa ('nao ha dados...'), veio: {result['answer']!r}"
        return True, result["answer"]
    return check


def mentions_years(*years: str):
    """Checker for qa_agent.ask(): final answer must mention all given years
    (e.g. a compound question spanning two periods must not silently drop one)."""
    def check(result) -> tuple[bool, str]:
        missing = [y for y in years if y not in result["answer"]]
        if missing:
            return False, f"faltando ano(s) {missing!r} em: {result['answer']!r}"
        return True, result["answer"]
    return check


def via_table_agent():
    """Checker for qa_agent.ask(): must have been answered by the Table
    Agent node (score == 1.0 is that node's fixed sentinel), not the RAG path."""
    def check(result) -> tuple[bool, str]:
        score = result["context"][0]["score"] if result["context"] else None
        if score != 1.0:
            return False, f"esperava roteamento pro Table Agent (score=1.0), veio score={score}"
        return True, f"score=1.0, resposta: {result['answer']!r}"
    return check


def via_rag_with_content():
    """Checker for qa_agent.ask(): must have gone through real vector RAG
    (score != 1.0, i.e. NOT the Table Agent's fixed sentinel) and produced
    a non-empty answer."""
    def check(result) -> tuple[bool, str]:
        score = result["context"][0]["score"] if result["context"] else None
        if score == 1.0:
            return False, "esperava RAG de verdade (score != 1.0), veio via Table Agent (score=1.0)"
        if not result["answer"].strip():
            return False, "resposta vazia"
        return True, f"score={score}, resposta: {result['answer']!r}"
    return check


# ---------------------------------------------------------------------------
# A) + B) Table Agent cases - run against src.table_agent.answer directly.
# ---------------------------------------------------------------------------

TABLE_AGENT_CASES = [
    # --- A) correct deterministic lookups: one per series, "mais recente"
    # variations, and explicit periods (month+year / quarter+year / bare year
    # for annual series) ---
    ("Qual foi o IPCA mais recente?", contains("%")),
    ("Qual a taxa de inflacao (IPCA) de janeiro de 2023?", contains("janeiro de 2023")),
    ("Qual foi o IPCA em dezembro de 1994?", contains("dezembro de 1994", "1.71")),  # regression: exact period, not "most recent"
    ("Qual foi o PIB mais recente?", contains("%")),
    ("Qual foi o PIB no 3 trimestre de 1999?", contains("1999")),
    ("Qual foi a taxa de desocupacao mais recente?", contains("%")),
    ("Como esta a taxa de desemprego no Brasil?", contains("PNAD")),
    ("Qual foi a taxa de desocupacao de maio de 2025?", contains("maio de 2025", "6.2")),  # regression: exact period, not "most recent"
    ("Qual foi o rendimento medio mais recente?", contains("R$")),
    ("Qual foi o rendimento medio de março de 2023?", contains("março de 2023")),
    ("Qual foi o salario medio dos brasileiros no ultimo trimestre movel?", contains("R$")),
    ("Qual a taxa de informalidade atual no mercado de trabalho?", contains("%")),
    ("Qual foi a taxa de informalidade no trimestre movel encerrado em janeiro de 2024?", contains("janeiro de 2024")),
    ("Qual foi a taxa de pobreza internacional mais recente?", contains("pobreza internacional")),
    ("Qual foi a taxa de pobreza internacional em 2020?", contains("pobreza internacional", "2020")),
    ("Qual foi a taxa de pobreza nacional mais recente?", contains("pobreza nacional")),
    ("Qual foi a taxa de pobreza nacional em 2023?", contains("pobreza nacional", "2023")),
    # regression: "internacional" not glued to "pobreza" as one exact phrase
    # must still win over the bare "pobreza" keyword (which would otherwise
    # silently answer with the NACIONAL series instead)
    ("Qual foi a taxa de pobreza pela linha internacional em 2015?", contains("pobreza internacional", "2015")),
    # regression: explicit PME mention must win over the "desemprego" keyword
    # shared with PNAD Continua, and the answer must not silently swap in
    # the national number instead
    ("Qual foi a taxa de desemprego da PME em 2010?", lambda r: (
        r is not None and "PME" in r["answer"] and "2010" in r["answer"] and "PNAD" not in r["answer"],
        r["answer"] if r else "None",
    )),
    ("Qual foi a taxa de desemprego media da PME mais recente?", contains("PME")),

    # --- B) deliberate refusals: must return None, never guess ---
    ("Qual foi a taxa de desocupacao em 2024?", is_none()),  # has data, but bare year is ambiguous for a moving-quarter series
    ("Qual foi o PIB em 1999?", is_none()),  # bare year ambiguous for a quarterly series
    ("Qual foi o rendimento medio em 2020?", is_none()),  # bare year ambiguous for a moving-quarter series
    ("Qual foi a taxa de desocupacao em janeiro de 2010?", is_none()),  # PNAD Continua only starts 2012-03
    ("Qual foi a taxa de desocupacao em dezembro de 2011?", is_none()),  # still before PNAD Continua's start
    ("Qual foi a taxa de desocupacao em 1995?", is_none()),  # no data at all for this series
    ("Qual foi a taxa de informalidade em 2005?", is_none()),  # out of range + ambiguous
    ("Qual foi o IPCA em janeiro de 1998 e de 1999?", is_none()),  # two periods named, one lookup can't answer both
    ("Qual foi o PIB no 1 trimestre de 2019 e no 2 trimestre de 2020?", is_none()),  # two periods named
    ("Qual a capital da Franca?", is_none()),  # out of domain entirely
    ("Qual o preco do Bitcoin hoje?", is_none()),  # out of domain entirely
    ("Qual o resultado do jogo do Brasil ontem?", is_none()),  # out of domain entirely
    ("Como a taxa de desocupacao varia por nivel de instrucao?", is_none()),  # names a series but wants descriptive prose, not a bare value
    ("Por que a taxa de desocupacao caiu no ultimo trimestre?", is_none()),  # analytical/causal, not a lookup
    ("Quais fatores explicam a alta do IPCA em 2021?", is_none()),  # analytical/causal, not a lookup
    ("Como o rendimento medio se distribui por regiao no Brasil?", is_none()),  # analytical/descriptive, not a lookup
]


# ---------------------------------------------------------------------------
# C) Full pipeline cases - run against src.qa_agent.ask (Table Agent routing
# + RAG fallback + LLM synthesis, all live).
# ---------------------------------------------------------------------------

QA_AGENT_CASES = [
    # simple lookups must still be routed to the Table Agent INSIDE the full
    # graph (score == 1.0 is try_table_agent's fixed sentinel)
    ("Qual foi o IPCA mais recente?", via_table_agent()),
    ("Como esta a taxa de desemprego no Brasil?", via_table_agent()),
    ("Qual foi o rendimento medio mais recente?", via_table_agent()),
    ("Qual foi a taxa de pobreza internacional mais recente?", via_table_agent()),
    ("Qual foi a taxa de desocupacao de maio de 2025?", lambda r: (
        r["context"][0]["score"] == 1.0 and "maio de 2025" in r["answer"],
        f"score={r['context'][0]['score']}, resposta={r['answer']!r}",
    )),
    ("Qual foi a taxa de desemprego da PME em 2010?", lambda r: (
        r["context"][0]["score"] == 1.0 and "PME" in r["answer"] and "2010" in r["answer"],
        f"score={r['context'][0]['score']}, resposta={r['answer']!r}",
    )),

    # revision lookups go to the Revision Agent (fixed score 1.0), not RAG or
    # the plain GDP lookup
    ("O PIB do 2 trimestre de 2022 foi revisado?", lambda r: (
        r["context"][0]["score"] == 1.0 and "3.2%" in r["answer"] and "3.5%" in r["answer"],
        f"resposta={r['answer']!r}",
    )),
    ("O PIB do 2 trimestre de 2020 sofreu revisao?", lambda r: (
        r["context"][0]["score"] == 1.0 and "-11.4%" in r["answer"] and "para cima" in r["answer"],
        f"resposta={r['answer']!r}",
    )),

    # analytical/out-of-range/out-of-domain questions the Table Agent
    # declines must, in the full pipeline, come back admitting lack of
    # data - never a fabricated explanation
    ("Por que a taxa de desocupacao caiu no ultimo trimestre?", has_decline_phrase()),
    ("Quais fatores explicam a alta do IPCA em 2021?", has_decline_phrase()),
    ("Como o rendimento medio se distribui por regiao no Brasil?", has_decline_phrase()),
    ("Qual a capital da Franca?", has_decline_phrase()),
    ("Qual o preco do Bitcoin hoje?", has_decline_phrase()),
    ("Qual foi o PIB em 1999?", has_decline_phrase()),
    ("Qual foi a taxa de desocupacao em 1995?", has_decline_phrase()),

    # a compound two-period question the Table Agent declines must, via
    # RAG (which can pull multiple chunks), mention BOTH periods - not
    # silently answer only one
    ("Qual foi o IPCA em janeiro de 1998 e de 1999?", mentions_years("1998", "1999")),

    # a descriptive question the PNAD data genuinely supports must produce
    # a real, non-empty RAG answer (score != 1.0 proves it did NOT come
    # from the Table Agent's fixed-score shortcut)
    ("Como a taxa de desocupacao varia por nivel de instrucao no Brasil?", via_rag_with_content()),
]


def _run(cases, call, label):
    passed = 0
    print(f"\n=== {label} ({len(cases)} casos) ===")
    for question, check in cases:
        try:
            result = call(question)
            ok, detail = check(result)
        except Exception as e:
            ok, detail = False, f"EXCEPTION: {e!r}"
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] {question}")
        print(f"       -> {detail}")
    return passed, len(cases)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    p1, t1 = _run(TABLE_AGENT_CASES, table_answer, "Table Agent")
    p2, t2 = _run(QA_AGENT_CASES, qa_ask, "QA Agent (pipeline completo)")

    total_passed, total = p1 + p2, t1 + t2
    print(f"\n=== RESUMO ===")
    print(f"Table Agent: {p1}/{t1} passaram")
    print(f"QA Agent:    {p2}/{t2} passaram")
    print(f"TOTAL:       {total_passed}/{total} passaram")

    if total_passed < total:
        raise SystemExit(1)
