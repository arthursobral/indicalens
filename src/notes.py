"""Week 2 ingestion: real prose from the IBGE PNAD Continua quarterly
"Comentarios" section, chunked and embedded like src/ingest.py's tabular
data. Source: the "Indicadores IBGE" booklet PDF published each quarter on
ftp.ibge.gov.br (agenciadenoticias.ibge.gov.br and apisidra.ibge.gov.br sit
behind a Cloudflare bot challenge that blocks plain HTTP clients; the FTP
mirror does not).

Usage:
    python -m src.notes
"""

import io
import re
import sys
from datetime import date

import requests
from pypdf import PdfReader

from src import db
from src.embeddings import embed
from src.ibge_client import format_period

BASE_URL = (
    "https://ftp.ibge.gov.br/Trabalho_e_Rendimento/"
    "Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/"
    "Fasciculos_Indicadores_IBGE"
)

_CHUNK_TARGET_CHARS = 700


def latest_caderno() -> tuple[str, str]:
    """Lists the FTP index to find the most recent quarterly booklet PDF.
    Returns (url, period) where period is an IBGE-style "YYYYQQ" code.
    """
    year = date.today().year
    for y in (year, year - 1):
        resp = requests.get(f"{BASE_URL}/{y}/", timeout=30)
        if resp.status_code != 200:
            continue
        matches = re.findall(r'href="(pnadc_(\d{6})_trimestre_caderno\.pdf)"', resp.text)
        if matches:
            filename, period = sorted(matches, key=lambda m: m[1])[-1]
            return f"{BASE_URL}/{y}/{filename}", period
    raise RuntimeError("no caderno PDF found on ftp.ibge.gov.br in the last two years")


def _is_header(line: str) -> bool:
    """Heuristic for a section title in the booklet: short, all-caps, at
    least two real words (rejects table/chart artifacts like '1oT 2oT ...').
    """
    if not (8 <= len(line) <= 90):
        return False
    letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", line)
    if len(letters) < 5 or letters != letters.upper():
        return False
    long_words = [w for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", line) if len(w) >= 4]
    return len(long_words) >= 2


def extract_sections(pdf_bytes: bytes) -> list[tuple[str, str]]:
    """Splits the booklet's "Comentarios" section into (header, body) pairs.
    A header can wrap across two extracted lines; merged when no body text
    has appeared yet between them.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() for page in reader.pages)
    start = full_text.find("Comentários")
    if start == -1:
        return []
    lines = [line.strip() for line in full_text[start:].split("\n") if line.strip()]

    sections: list[tuple[str, str]] = []
    header, body = None, []
    for line in lines:
        if _is_header(line):
            if header and not body:
                header = f"{header} {line}"
                continue
            if header:
                sections.append((header, " ".join(body)))
            header, body = line, []
        elif header:
            body.append(line)
    if header:
        sections.append((header, " ".join(body)))
    return sections


def _split_into_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > _CHUNK_TARGET_CHARS:
            chunks.append(current.strip())
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _slug(header: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", header).strip("_").upper()[:50]


def build_rows() -> list[dict]:
    url, period = latest_caderno()
    pdf_bytes = requests.get(url, timeout=60).content
    sections = extract_sections(pdf_bytes)
    when = format_period("quarterly", period)

    rows = []
    for header, body in sections:
        title = header.title()
        for i, chunk in enumerate(_split_into_chunks(body)):
            rows.append({
                "source": "IBGE-PNAD-COMENTARIOS",
                "series": f"PNAD_COMENTARIO_{_slug(header)}",
                "period": f"{period}_{i}",
                "content": (
                    f"Segundo os comentarios do IBGE sobre a PNAD Continua "
                    f"({when}), a respeito de {title}: {chunk}"
                ),
                "citation": (
                    f"IBGE/PNAD Continua - Comentarios - {title} - {when} "
                    f"(Indicadores IBGE, caderno trimestral)"
                ),
            })
    return rows


def run() -> int:
    rows = build_rows()
    embeddings = embed([r["content"] for r in rows])
    for row, vec in zip(rows, embeddings):
        row["embedding"] = vec

    with db.connect() as conn:
        db.upsert_documents(conn, rows)

    return len(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    n = run()
    print(f"Ingeridos/atualizados {n} documentos de comentarios.")
