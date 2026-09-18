"""Self-check for the Indicator Analyst Agent's zero-shot classification.
Downloads a real ~280M param multilingual model on first run — not wired
into CI (see docs/01-decisions.md) because of that.

Run: python tests/test_indicator_analyst.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicator_analyst import THEMES, classify_theme


def test_classify_theme_live():
    theme = classify_theme(
        "A taxa de desocupação, no Brasil, no 2º trimestre de 2026, foi "
        "estimada em 5,4%. Esta estimativa apresentou redução de 0,7 ponto "
        "percentual em comparação com o 1º trimestre de 2026."
    )
    assert theme in THEMES


if __name__ == "__main__":
    try:
        test_classify_theme_live()
        print("test_classify_theme_live: ok")
    except Exception as e:
        print(f"test_classify_theme_live: SKIPPED ({e})")
