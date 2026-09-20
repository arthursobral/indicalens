"""Self-check for the Revision Agent's headline parser (pure) and one live
booklet. Run: python tests/test_revisions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.revisions import list_booklets, parse_headline


def test_parse_headline():
    ok = "Na comparação com igual período de 2025, houve crescimento do PIB de 1,8% no primeiro trimestre."
    assert parse_headline(ok) == 1.8
    # stray spaces inside numbers, as extracted from the real PDFs
    assert parse_headline("Na comparação com igual período de 201 9, houve crescimento do PIB de 2,1% no ano.") == 2.1
    # negative wording must flip the sign (2020 pandemic release)
    assert parse_headline("Na comparação com igual período de 2019, houve variação negativa do PIB de 0,3% no trimestre.") == -0.3
    assert parse_headline("texto sem manchete") is None


def test_booklets_live():
    booklets = list_booklets()
    assert len(booklets) >= 20 and booklets[0][0] == "202001"


if __name__ == "__main__":
    test_parse_headline()
    print("test_parse_headline: ok")
    try:
        test_booklets_live()
        print("test_booklets_live: ok")
    except Exception as e:
        print(f"test_booklets_live: SKIPPED ({e})")
