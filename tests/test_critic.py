"""Self-check for the Critic Agent. Loads a ~280M param NLI model, so it is
not wired into CI (same reasoning as test_indicator_analyst.py).

Run: python tests/test_critic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.critic import split_claims, verify

CTX = [
    {"content": "Segundo o IBGE (SIDRA, tabela 1737), IPCA - variação mensal em janeiro de 1998 foi de 0.71%."},
    {"content": "Segundo o IBGE (SIDRA, tabela 1737), IPCA - variação mensal em janeiro de 1999 foi de 0.70%."},
]


def test_split_claims():
    claims = split_claims("Resumo:\nO IPCA em janeiro de 1998 foi de **0,71 %** [1].\nEm 1999, foi de 0,70 % [2].")
    assert [c[1] for c in claims] == [[1], [2]]  # lead-in "Resumo:" dropped, citations kept aside
    assert "[" not in claims[0][0] and "*" not in claims[0][0]


def test_verify_live():
    good = verify("O IPCA em janeiro de 1998 foi de 0,71 % [1].", CTX)
    assert [v["status"] for v in good] == ["supported"]

    wrong = verify("O IPCA em janeiro de 1998 foi de 0,95 % [1].", CTX)
    assert [v["status"] for v in wrong] == ["contradicted"]

    invented = verify("O IPCA em janeiro de 1998 foi de 0,71 % [1]. Foi o maior da década por causa da crise asiática.", CTX)
    assert [v["status"] for v in invented] == ["supported", "unsupported"]

    assert verify("Não há dados suficientes ingeridos para responder.", CTX) == []


if __name__ == "__main__":
    test_split_claims()
    print("test_split_claims: ok")
    test_verify_live()
    print("test_verify_live: ok")
