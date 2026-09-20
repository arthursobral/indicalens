"""Self-check for the Correlation Agent (synthetic data, hermetic; runs in CI).
Run: python tests/test_correlation.py [--live]
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import correlation as c


def _months(n, start="200001"):
    return [c.add_months(start, i) for i in range(n)]


def test_add_months():
    assert c.add_months("200001", -1) == "199912" and c.add_months("200012", 1) == "200101"


def test_recovers_planted_lag():
    random.seed(1)
    ms = _months(200)
    x = {m: random.gauss(0, 1) for m in ms}
    y = {m: x.get(c.add_months(m, -3), 0) + random.gauss(0, 0.3) for m in ms}  # x leads y by 3
    res = c.analyze(x, y)
    assert res["best"]["lag"] == 3 and res["best"]["r"] > 0.9 and res["best"]["distinguishable"]


def test_independent_noise_is_not_distinguishable():
    random.seed(2)
    ms = _months(200)
    x = {m: random.gauss(0, 1) for m in ms}
    y = {m: random.gauss(0, 1) for m in ms}
    res = c.analyze(x, y)
    assert not res["best"]["distinguishable"], res["best"]


def test_ci_matches_known_value():
    lo, hi = c._ci(0.5, 103, 0.05)  # Fisher z: atanh(.5)=.5493, se=.1, +-.196 -> tanh -> [.3393, .6323]
    assert abs(lo - 0.3393) < 1e-3 and abs(hi - 0.6323) < 1e-3, (lo, hi)


def test_random_walk_levels_mislead_but_variations_do_not():
    ms = _months(300)
    levels = diffs = 0
    for seed in range(60):  # two INDEPENDENT random walks, many draws
        random.seed(seed)
        w1, w2, a, b = {}, {}, 0.0, 0.0
        for m in ms:
            a += random.gauss(0, 1)
            b += random.gauss(0, 1)
            w1[m], w2[m] = a, b
        levels += c.analyze(w1, w2)["best"]["distinguishable"]
        diffs += c.analyze(c._diff(w1), c._diff(w2))["best"]["distinguishable"]
    assert levels / 60 > 0.6, levels  # levels: spurious "relation" most of the time (measured 0.85)
    assert diffs / 60 < 0.2, diffs  # variations: near the nominal 5% (measured 0.07)


def test_detect():
    assert c.detect("A inflacao se move junto com a Selic?") == ("SELIC_MENSAL", "ipca")
    assert c.detect("O câmbio afeta o IPCA?") == ("CAMBIO_USD_MEDIA_MENSAL", "ipca")
    assert c.detect("O dólar influencia a taxa de desocupação?")[1] == "outro"
    assert c.detect("Qual foi o IPCA em agosto de 2026?") is None
    assert c.detect("Qual a Selic hoje?") is None
    res = c.analyze({m: float(i % 5) for i, m in enumerate(_months(100))}, {m: float((i * 3) % 7) for i, m in enumerate(_months(100))})
    assert "câmbio" in c.describe("CAMBIO_USD_MEDIA_MENSAL", res) and "Selic também" not in c.describe("CAMBIO_USD_MEDIA_MENSAL", res)
    assert "vou estimar" in c.answer("O dólar influencia a taxa de desocupação?")["answer"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    if "--live" in sys.argv:
        r = c.answer("A inflacao se move junto com a Selic?")
        print(r["answer"])
