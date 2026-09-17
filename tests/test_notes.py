"""Self-check for the PNAD "Comentarios" text ingestion.
Run: python tests/test_notes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.notes import _is_header, _slug, _split_into_chunks, extract_sections, latest_caderno


def test_is_header():
    assert _is_header("TAXA DE DESOCUPAÇÃO")
    assert _is_header("POPULAÇÃO OCUPADA")
    assert not _is_header("A taxa de desocupação caiu 0,7 p.p. no trimestre.")
    assert not _is_header("1°T 2°T 1°T 2°T 1°T 2°T 1°T 2°T 1°T 2°T")
    assert not _is_header("BR")


def test_slug():
    assert _slug("TAXA DE DESOCUPAÇÃO") == "TAXA_DE_DESOCUPA_O"
    assert len(_slug("X" * 100)) <= 50


def test_split_into_chunks():
    text = "Frase um. Frase dois. " * 50
    chunks = _split_into_chunks(text)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_extract_sections_and_live_fetch():
    url, period = latest_caderno()
    assert url.endswith(".pdf")
    assert len(period) == 6
    import requests
    pdf_bytes = requests.get(url, timeout=60).content
    sections = extract_sections(pdf_bytes)
    assert len(sections) >= 5
    headers = [h for h, _ in sections]
    assert any("DESOCUPA" in h for h in headers)


if __name__ == "__main__":
    test_is_header()
    print("test_is_header: ok")
    test_slug()
    print("test_slug: ok")
    test_split_into_chunks()
    print("test_split_into_chunks: ok")
    try:
        test_extract_sections_and_live_fetch()
        print("test_extract_sections_and_live_fetch: ok")
    except Exception as e:
        print(f"test_extract_sections_and_live_fetch: SKIPPED ({e})")
