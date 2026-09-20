"""Self-check for the vintage parser and the revision-vs-market analysis (synthetic, hermetic; runs in CI).
Run: python tests/test_revision_effects.py
"""

import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import revision_effects as fx
from src import revisions as rv


def test_parse_vintage_table_keeps_neighbouring_numbers_apart():
    text = ("Taxas (%) 2022.IV 2023.I 2023.II 2023.III 2023.IV Trimestre / mesmo trimestre do ano\n anterior\n"
            " < Anexo: Tabela 2 >\n 2,7 4,2 3,5 2,0 2,1 Trimestre / trimestre imediatamente anterior").replace("\n", "\n")
    assert rv.parse_vintage_table(text) == {"202204": 2.7, "202301": 4.2, "202302": 3.5, "202303": 2.0, "202304": 2.1}
    assert rv.parse_vintage_table("nothing here") is None


def test_release_date():
    assert rv.parse_release_date("Publicado em 29/05/2020 às 9 horas") == "2020-05-29"
    assert rv.parse_release_date("Atualizado em 01/03/2024 às 09 horas") == "2024-03-01"
    assert rv.parse_release_date("sem data") is None


def _vint():
    # three releases; Q1 first 1.0, revised to 1.5 in the next release and 2.0 in the one after
    return [
        {"period": "202001", "released": "2020-05-29", "table": {"201901": 0.5, "201902": 0.6, "201903": 0.7, "201904": 0.8, "202001": 1.0}},
        {"period": "202002", "released": "2020-09-01", "table": {"201902": 0.6, "201903": 0.7, "201904": 0.8, "202001": 1.5, "202002": -9.0}},
        {"period": "202003", "released": "2020-12-03", "table": {"201903": 0.7, "201904": 0.9, "202001": 2.0, "202002": -8.0, "202003": -3.0}},
    ]


def test_same_age_revisions_and_events():
    k1 = {r["period"]: r["revision"] for r in rv.same_age_revisions(_vint(), 1)}
    assert k1 == {"202001": 0.5, "202002": 1.0}  # value one release later minus first release
    assert {r["period"]: r["revision"] for r in rv.same_age_revisions(_vint(), 2)} == {"202001": 1.0}
    ev = fx.events(_vint())
    assert [e["news"] for e in ev] == [0.5, 1.6]  # release 2: only 202001 moved (+0.5); release 3: 0.1 + 0.5 + 1.0
    assert ev[0]["date"] == date(2020, 9, 1) and ev[0]["n_quarters"] == 4


def test_outcomes_use_last_value_on_or_before_and_drop_incomplete_windows():
    evs = [{"period": "x", "date": date(2020, 1, 10), "news": 1.0, "n_quarters": 4},
           {"period": "y", "date": date(2020, 6, 10), "news": 0.0, "n_quarters": 4}]
    selic = [(date(2019, 10, 1), 5.0), (date(2020, 1, 5), 4.5), (date(2020, 4, 1), 4.0), (date(2020, 6, 1), 3.0)]
    usd = [(date(2019, 10, 1), 4.0), (date(2020, 1, 9), 4.0), (date(2020, 4, 9), 4.4), (date(2020, 6, 9), 4.0)]
    out = fx.outcomes(evs, selic, usd, today=date(2020, 5, 1))  # the June event's 90-day window is still open
    assert out["selic_depois"] == [(1.0, -0.5)]  # 4.0 (Apr 9 -> last on/before) minus 4.5 (Jan 10)
    assert [round(v, 3) for _, v in out["dolar_depois"]] == [10.0]
    assert len(out["selic_antes"]) == 2  # placebo windows are complete


def test_stats_finds_planted_effect_and_not_noise():
    shape = [0.0] * 19 + [0.5, 0.6, 2.1, 1.8, 0.8, 0.5]
    random.seed(0)
    planted = [(x, 1.5 * x + random.gauss(0, 0.3)) for x in shape]
    assert fx.stats(planted, perms=2000)["p_perm"] < 0.01
    hits = sum(fx.stats([(x, random.gauss(0, 1)) for x in shape], perms=300, seed=i)["p_perm"] < 0.05 for i in range(40))
    assert hits <= 6  # nominal 5% of 40 = 2; loose bound so the test is not flaky


def test_matches():
    assert fx.matches("as revisoes do pib afetam a selic?") and fx.matches("a revisao do pib move o dolar?")
    assert not fx.matches("o pib de 2022 foi revisado?") and not fx.matches("a inflacao se move junto com a selic?")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
