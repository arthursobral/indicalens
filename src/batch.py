"""Batch: one report per indicator, each answering a fixed question set through the whole
graph (deterministic agents first, RAG + Critic for commentary), with time, LLM tokens and
Critic flags per report. A failing question is recorded and the batch goes on.

Usage:
    python -m src.batch [--only IPCA,PIB] [--out reports]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src import critic, llm, qa_agent

REPORTS = {
    "IPCA": ["Qual foi o IPCA mais recente?", "A inflacao se move junto com a Selic?", "O cambio afeta o IPCA?"],
    "PIB": ["Qual foi o PIB mais recente?", "O PIB do 2 trimestre de 2022 foi revisado?"],
    "Desocupacao": ["Qual foi a taxa de desocupacao mais recente?", "O que os comentarios do IBGE dizem sobre a desocupacao?"],
    "Rendimento": ["Qual foi o rendimento medio mais recente?", "O que os comentarios do IBGE dizem sobre o rendimento?"],
    "Informalidade": ["Qual foi a taxa de informalidade mais recente?", "O que os comentarios do IBGE dizem sobre a informalidade?"],
    "Pobreza": ["Qual foi a taxa de pobreza internacional mais recente?", "Qual foi a taxa de pobreza nacional mais recente?"],
}
RETRIES = 2  # Groq's free tier answers 429 under load; one retry after a pause is enough in practice


def ask_once(question: str, ask=qa_agent.ask, sleep=time.sleep) -> dict:
    """One question through the graph: {answer, path, seconds, error, flagged, inferred, claims}."""
    start = time.time()
    for attempt in range(1, RETRIES + 1):
        try:
            state = ask(question)
            break
        except Exception as e:  # noqa: BLE001 - a batch must survive any single failure
            if attempt == RETRIES:
                return {"question": question, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - start, 2)}
            sleep(5)
    verdicts = state["verdicts"]
    deterministic = bool(state["context"]) and all(c.get("score") == 1.0 for c in state["context"])
    return {
        "question": question,
        "answer": state["answer"],
        "sources": [c["citation"] for c in state["context"]],
        "path": "deterministic" if deterministic else "rag",
        "seconds": round(time.time() - start, 2),
        "claims": len(verdicts),
        "flagged": sum(v["status"] in critic.HARD_FLAGS for v in verdicts),
        "inferred": sum(v["status"] == "inferred" for v in verdicts),
    }


def build_report(name: str, questions: list[str], ask=qa_agent.ask, sleep=time.sleep) -> dict:
    before, start = dict(llm.USAGE), time.time()
    items = [ask_once(q, ask, sleep) for q in questions]
    return {
        "indicator": name,
        "items": items,
        "seconds": round(time.time() - start, 2),
        "llm_calls": llm.USAGE["calls"] - before["calls"],
        "tokens_in": llm.USAGE["input"] - before["input"],
        "tokens_out": llm.USAGE["output"] - before["output"],
        "errors": sum("error" in i for i in items),
        "flagged": sum(i.get("flagged", 0) for i in items),
        "claims": sum(i.get("claims", 0) for i in items),
    }


def to_markdown(r: dict) -> str:
    lines = [f"# Relatório: {r['indicator']}", "",
             f"_{len(r['items'])} perguntas, {r['seconds']}s, {r['llm_calls']} chamadas ao LLM "
             f"({r['tokens_in']} tokens de entrada, {r['tokens_out']} de saída), "
             f"{r['flagged']} de {r['claims']} afirmações sinalizadas pelo Critic, {r['errors']} erro(s)._", ""]
    for i in r["items"]:
        lines += [f"## {i['question']}", ""]
        if "error" in i:
            lines += [f"**Erro:** {i['error']}", ""]
            continue
        lines += [i["answer"], ""]
        if "Fontes:" not in i["answer"]:  # the correlation text already lists its sources
            lines += ["Fontes: " + ("; ".join(i["sources"]) if i["sources"] else "nenhuma recuperada"), ""]
    return "\n".join(lines)


def run(only: list[str] | None, out: Path, ask=qa_agent.ask, sleep=time.sleep) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    start, reports = time.time(), []
    for name, questions in REPORTS.items():
        if only and name not in only:
            continue
        r = build_report(name, questions, ask, sleep)
        (out / f"{name}.md").write_text(to_markdown(r), encoding="utf-8")
        reports.append(r)
    n = len(reports)
    elapsed = time.time() - start
    summary = {
        "reports": n,
        "seconds": round(elapsed, 2),
        "seconds_per_report": round(elapsed / n, 2) if n else 0,
        "tokens_in_per_report": round(sum(r["tokens_in"] for r in reports) / n) if n else 0,
        "tokens_out_per_report": round(sum(r["tokens_out"] for r in reports) / n) if n else 0,
        "errors": sum(r["errors"] for r in reports),
        "per_report": [{k: v for k, v in r.items() if k != "items"} for r in reports],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated indicators, e.g. IPCA,PIB")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    s = run(args.only.split(",") if args.only else None, Path(args.out))
    print(f"\n{'relatorio':<15}{'s':>7}{'LLM':>5}{'tok in':>8}{'tok out':>8}{'critic':>9}{'erros':>7}")
    for r in s["per_report"]:
        print(f"{r['indicator']:<15}{r['seconds']:>7}{r['llm_calls']:>5}{r['tokens_in']:>8}{r['tokens_out']:>8}"
              f"{str(r['flagged']) + '/' + str(r['claims']):>9}{r['errors']:>7}")
    print(f"\n{s['reports']} relatorios em {s['seconds']}s ({s['seconds_per_report']}s cada); "
          f"tokens por relatorio: {s['tokens_in_per_report']} entrada / {s['tokens_out_per_report']} saida; erros: {s['errors']}")
