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

from src import bcb_agent, correlation, critic, db, revisions, table_agent
from src.embeddings import embed_one
from src.llm import chat
from src.tracing import flush, observe

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


@observe(name="try_table_agent")
def try_table_agent(state: State) -> dict:
    if state["answer"]:  # already answered by the revision lookup
        return {}
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


@observe(name="try_revision_agent")
def try_revision_agent(state: State) -> dict:
    try:
        result = revisions.answer(state["question"])
    except Exception:
        return {}
    if result is None:
        return {}
    return {
        "answer": result["answer"],
        "context": [{"content": result["answer"], "citation": result["citation"], "theme": None, "score": 1.0}],
    }


def _try_agent(state: State, agent) -> dict:
    if state["answer"]:
        return {}
    try:
        result = agent.answer(state["question"])
    except Exception:  # live IBGE/BCB failure: fall back instead of crashing
        return {}
    if result is None:
        return {}
    return {
        "answer": result["answer"],
        "context": [{"content": result["answer"], "citation": result["citation"], "theme": None, "score": 1.0}],
    }


@observe(name="try_correlation_agent")
def try_correlation_agent(state: State) -> dict:
    return _try_agent(state, correlation)


@observe(name="try_bcb_agent")
def try_bcb_agent(state: State) -> dict:
    return _try_agent(state, bcb_agent)


def _route_after_table_agent(state: State) -> str:
    return "done" if state["answer"] else "rag"


@observe(name="retrieve")
def retrieve(state: State) -> dict:
    query_vec = embed_one(state["question"])
    with db.connect() as conn:
        context = db.retrieve(conn, query_vec, k=5)
    return {"context": context}


@observe(name="generate")
def generate(state: State) -> dict:
    numbered = "\n".join(
        f"[{i + 1}] {c['content']} (fonte: {c['citation']})"
        for i, c in enumerate(state["context"])
    )
    user_prompt = f"Contexto:\n{numbered}\n\nPergunta: {state['question']}"
    answer = chat(SYSTEM_PROMPT, user_prompt)
    if not answer.strip():  # the LLM occasionally returns an empty completion (seen by the eval harness)
        answer = chat(SYSTEM_PROMPT, user_prompt)
    if not answer.strip():
        answer = "Não foi possível gerar uma resposta agora. Tente novamente."
    return {"answer": answer}


@observe(name="verify")
def verify(state: State) -> dict:
    verdicts = critic.verify(state["answer"], state["context"])
    flagged = [v for v in verdicts if v["status"] in critic.HARD_FLAGS]
    inferred = [v for v in verdicts if v["status"] == "inferred"]
    answer = state["answer"]
    if flagged:
        lines = "\n".join(f"- {v['claim']} ({v['status']})" for v in flagged)
        answer += f"\n\n[Critic] Afirmações não sustentadas pelas fontes recuperadas:\n{lines}"
    if inferred:
        lines = "\n".join(f"- {v['claim']}" for v in inferred)
        answer += f"\n\n[Critic] Conclusões inferidas pelo modelo (não escritas na fonte):\n{lines}"
    return {"verdicts": verdicts, "answer": answer}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("try_revision_agent", try_revision_agent)
    graph.add_node("try_correlation_agent", try_correlation_agent)
    graph.add_node("try_bcb_agent", try_bcb_agent)
    graph.add_node("try_table_agent", try_table_agent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_edge(START, "try_revision_agent")
    graph.add_edge("try_revision_agent", "try_correlation_agent")
    graph.add_edge("try_correlation_agent", "try_bcb_agent")
    graph.add_edge("try_bcb_agent", "try_table_agent")
    graph.add_conditional_edges("try_table_agent", _route_after_table_agent, {"done": END, "rag": "retrieve"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


@observe(name="ask")
def ask(question: str) -> State:
    result = build_graph().invoke({"question": question, "context": [], "answer": "", "verdicts": []})
    flush()
    return result


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
