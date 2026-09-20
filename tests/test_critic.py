"""Self-check for the Critic Agent. Loads a ~280M param NLI model, so it is
not wired into CI (same reasoning as test_indicator_analyst.py).

Run: python tests/test_critic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.critic import HARD_FLAGS, _is_inference, is_short_numeric, split_claims, verify, verify_claim

CTX = [
    {"content": "Segundo o IBGE (SIDRA, tabela 1737), IPCA - variação mensal em janeiro de 1998 foi de 0.71%."},
    {"content": "Segundo o IBGE (SIDRA, tabela 1737), IPCA - variação mensal em janeiro de 1999 foi de 0.70%."},
]


def test_split_claims():
    claims = split_claims("Resumo:\nO IPCA em janeiro de 1998 foi de **0,71 %** [1].\nEm 1999, foi de 0,70 % [2].")
    assert [c[1] for c in claims] == [[1], [2]]  # lead-in "Resumo:" dropped, citations kept aside
    assert "[" not in claims[0][0] and "*" not in claims[0][0]


def test_split_claims_skips_non_claims():
    answer = "\n".join([
        "| Região | Taxa (%) |",
        "|---|---|",
        "| Nordeste | 7,6 % |",
        "Fonte: IBGE/PNAD Contínua – Comentários – 2.º trimestre de 2026.",
        "Composição por cor ou raça",
        "Se precisar de dados de anos anteriores, informe-me!",
    ])
    assert [c[0] for c in split_claims(answer)] == ["Nordeste: 7,6 %"]


def test_short_numeric_claims_are_checked_lexically():
    ctx = [{"content": "A Região Nordeste registrou uma taxa de 7,6%; enquanto a Região Sul teve a menor, 3,2%."}]
    assert is_short_numeric("Nordeste: 7,6 %") and not is_short_numeric("A taxa de desocupação tende a diminuir com a instrução")
    assert verify_claim("Nordeste: 7,6 %", [], ctx)["status"] == "supported"  # 7,6 == 7.60, no model needed
    assert verify_claim("Nordeste: 9,9 %", [], ctx)["status"] == "unsupported"  # number not in the source


def test_inference_category():
    ctx = [{"content": "Com nível superior completo o nível da ocupação chegou a 80,2%; sem instrução, 21,3%."}]
    # a conclusion that only reuses numbers already in the source is an inference, not an error
    assert _is_inference("Portanto, quanto maior a instrução, maior a ocupação (21,3% contra 80,2%).", ctx)
    # a wrong number can never hide behind an inference marker
    assert not _is_inference("Portanto, quanto maior a instrução, maior a ocupação (21,3% contra 95,0%).", ctx)
    # no marker, no downgrade
    assert not _is_inference("A ocupação chegou a 80,2%.", ctx)
    assert "inferred" not in HARD_FLAGS


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
    test_split_claims_skips_non_claims()
    print("test_split_claims_skips_non_claims: ok")
    test_short_numeric_claims_are_checked_lexically()
    print("test_short_numeric_claims_are_checked_lexically: ok")
    test_inference_category()
    print("test_inference_category: ok")
    test_verify_live()
    print("test_verify_live: ok")
