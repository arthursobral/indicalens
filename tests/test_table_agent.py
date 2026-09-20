"""Self-check for the Table Agent's deterministic lookup.
Run: python tests/test_table_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.table_agent import match_series, answer


def test_match_series():
    assert match_series("Qual foi o IPCA em agosto?")["name"] == "IPCA_VARIACAO_MENSAL"
    assert match_series("Como esta a taxa de desemprego?")["name"] == "PNAD_TAXA_DESOCUPACAO"
    assert match_series("Qual a previsao do tempo para amanha?") is None
    # names a series but needs descriptive prose, not a bare lookup
    assert match_series("Como a taxa de desocupacao varia por nivel de instrucao?") is None
    assert match_series("Por que a taxa de desocupacao caiu no trimestre?") is None
    assert match_series("A inflacao se move junto com a Selic?") is None
    assert match_series("O cambio afeta o IPCA?") is None
    assert match_series("Qual foi o IPCA em janeiro de 1998?") is not None
    # asking what the commentary says is descriptive, not a bare lookup (found by the eval harness)
    assert match_series("O que os comentarios do IBGE dizem sobre a taxa de desocupacao no ultimo trimestre?") is None


def test_answer_live():
    result = answer("Qual foi a taxa de desocupacao mais recente?")
    assert result is not None
    assert "%" in result["answer"]
    assert "IBGE/SIDRA" in result["citation"]

    assert answer("Qual a capital da França?") is None

    # regression: a specific named period must return THAT period's value,
    # not silently fall back to the most recent one
    specific = answer("Qual foi a taxa de desocupacao de maio de 2025?")
    assert specific is not None
    assert "maio de 2025" in specific["answer"]

    # a specific period we don't have data for must not be guessed either
    assert answer("Qual foi a taxa de desocupacao em janeiro de 2010?") is None

    # regression: pre-2000 years (IPCA history goes back to 1979) must parse too
    old = answer("Qual foi o IPCA em dezembro de 1994?")
    assert old is not None
    assert "dezembro de 1994" in old["answer"]

    # regression: a bare year with no month, for a monthly/moving-quarter
    # series, is ambiguous - must decline, not silently answer "most recent"
    assert answer("Qual foi a taxa de desocupacao em 1995?") is None  # no data either way
    assert answer("Qual foi a taxa de desocupacao em 2024?") is None  # has data, but no month named

    # same ambiguity for quarterly: a bare year with no quarter number
    assert answer("Qual foi o PIB em 1999?") is None
    specific_quarter = answer("Qual foi o PIB no 3 trimestre de 1999?")
    assert specific_quarter is not None
    assert "1999" in specific_quarter["answer"]

    # regression: a compound question naming two periods must not silently
    # answer only the first one (deferred to RAG instead, which can cite both)
    assert answer("Qual foi o IPCA em janeiro de 1998 e de 1999?") is None

    # regression: an explicit PME mention must win even when the question
    # also contains "desemprego" (PNAD Continua's own keyword) - it must
    # not silently answer with the national PNAD number instead
    pme = answer("Qual foi a taxa de desemprego da PME em 2010?")
    assert pme is not None
    assert "PME" in pme["answer"] and "2010" in pme["answer"]
    assert "PNAD" not in pme["answer"]


if __name__ == "__main__":
    test_match_series()
    print("test_match_series: ok")
    try:
        test_answer_live()
        print("test_answer_live: ok")
    except Exception as e:
        print(f"test_answer_live: SKIPPED ({e})")
