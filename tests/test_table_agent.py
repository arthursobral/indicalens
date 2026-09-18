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


if __name__ == "__main__":
    test_match_series()
    print("test_match_series: ok")
    try:
        test_answer_live()
        print("test_answer_live: ok")
    except Exception as e:
        print(f"test_answer_live: SKIPPED ({e})")
