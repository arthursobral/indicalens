"""QA agent: a LangGraph that first tries the Table Agent (deterministic
lookup against a matched IBGE series) and, when the question doesn't name a
known indicator, falls back to vector RAG (retrieve -> generate), grounded
in retrieved chunks and always citing its sources.

Usage:
    python -m src.qa_agent "Qual foi a taxa de desocupacao no ultimo trimestre?"
"""

import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src import critic, db, table_agent
from src.embeddings import embed_one
from src.llm import chat

SYSTEM_PROMPT = (
    "Voce e um analista que responde perguntas sobre indicadores do IBGE "
    "usando APENAS os trechos de contexto fornecidos, cada um marcado [n]. "
    "Cite a fonte de cada afirmacao numerica usando o marcador [n] correspondente. "
    "Se o contexto nao tiver a resposta, diga claramente que nao ha dados "
    "suficientes ingeridos para responder, em vez de inventar um numero."
)


class State(TypedDict):
    question: str
    context: list[dict]
    answer: str
    verdicts: list[dict]


def try_table_agent(state: State) -> dict:
    try:
        result = table_agent.answer(state["question"])
    except Exception:
        # table_agent hits the live IBGE API on every call; a transient
        # network/API failure here should fall back to RAG, not crash the
        # whole answer.
        return {}
    if result is None:
        return {}
    return {
        "answer": result["answer"],
        "context": [{"content": result["answer"], "citation": result["citation"], "theme": None, "score": 1.0}],
    }


def _route_after_table_agent(state: State) -> str:
    return "done" if state["answer"] else "rag"


def retrieve(state: State) -> dict:
    query_vec = embed_one(state["question"])
    with db.connect() as conn:
        context = db.retrieve(conn, query_vec, k=5)
    return {"context": context}


def generate(state: State) -> dict:
    numbered = "\n".join(
        f"[{i + 1}] {c['content']} (fonte: {c['citation']})"
        for i, c in enumerate(state["context"])
    )
    user_prompt = f"Contexto:\n{numbered}\n\nPergunta: {state['question']}"
    answer = chat(SYSTEM_PROMPT, user_prompt)
    return {"answer": answer}


def verify(state: State) -> dict:
    verdicts = critic.verify(state["answer"], state["context"])
    flagged = [v for v in verdicts if v["status"] != "supported"]
    answer = state["answer"]
    if flagged:
        lines = "\n".join(f"- {v['claim']} ({v['status']})" for v in flagged)
        answer += f"\n\n[Critic] Afirmações não sustentadas pelas fontes recuperadas:\n{lines}"
    return {"verdicts": verdicts, "answer": answer}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("try_table_agent", try_table_agent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_edge(START, "try_table_agent")
    graph.add_conditional_edges("try_table_agent", _route_after_table_agent, {"done": END, "rag": "retrieve"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def ask(question: str) -> State:
    return build_graph().invoke({"question": question, "context": [], "answer": "", "verdicts": []})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print('Uso: python -m src.qa_agent "sua pergunta"')
        raise SystemExit(1)

    result = ask(" ".join(sys.argv[1:]))
    print(result["answer"])
    if result["verdicts"]:
        ok = sum(v["status"] == "supported" for v in result["verdicts"])
        print(f"\nCritic: {ok}/{len(result['verdicts'])} afirmações sustentadas")
    print("\nFontes recuperadas:")
    for c in result["context"]:
        theme = f", tema={c['theme']}" if c.get("theme") else ""
        print(f"  - {c['citation']} (score={c['score']:.3f}{theme})")
