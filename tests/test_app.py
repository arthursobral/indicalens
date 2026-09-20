"""Self-check for the Streamlit UI with a stubbed graph (hermetic; runs in CI).
Run: python tests/test_app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

from src import batch

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _fake(question, *a, **k):
    return {"question": question, "answer": f"resposta para: {question}", "sources": ["fonte X"], "path": "rag",
            "seconds": 1.5, "claims": 3, "flagged": 1, "inferred": 0}


def _run(env: dict, questions: list[str]):
    old_env = {k: os.environ.get(k) for k in env}
    old_ask = batch.ask_once
    os.environ.update(env)
    batch.ask_once = _fake
    try:
        at = AppTest.from_file(APP, default_timeout=30).run()
        for q in questions:
            at.chat_input[0].set_value(q).run()
        return at
    finally:
        batch.ask_once = old_ask
        for k, v in old_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_missing_config_stops_with_a_clear_message():
    saved = {k: os.environ.pop(k, None) for k in ("DATABASE_URL", "GROQ_API_KEY", "OLLAMA_HOST")}
    try:
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert any("DATABASE_URL" in e.value for e in at.error)
        assert len(at.chat_input) == 0  # nothing to type into: the app stopped
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_question_is_answered_and_shown():
    at = _run({"DATABASE_URL": "postgresql://x", "GROQ_API_KEY": "x"}, ["Qual o IPCA?"])
    assert not at.exception
    texts = " ".join(m.value for m in at.markdown)
    assert "Qual o IPCA?" in texts and "resposta para: Qual o IPCA?" in texts
    chips = " ".join(m.value for m in at.markdown)
    assert "Critic: 1 de 3 afirmações sinalizadas" in chips and "Busca + LLM + Critic" in chips


def test_session_cap_blocks_extra_questions():
    old = os.environ.get("MAX_QUESTIONS_PER_SESSION")
    os.environ["MAX_QUESTIONS_PER_SESSION"] = "2"
    try:
        at = _run({"DATABASE_URL": "postgresql://x", "GROQ_API_KEY": "x"}, ["a", "b", "c"])
    finally:
        os.environ.pop("MAX_QUESTIONS_PER_SESSION") if old is None else os.environ.__setitem__("MAX_QUESTIONS_PER_SESSION", old)
    assert any("Limite de 2 perguntas" in w.value for w in at.warning)
    assert "resposta para: c" not in " ".join(m.value for m in at.markdown)  # the third never reached the LLM


def test_critic_switch_is_explicit():
    from src import qa_agent

    old = os.environ.get("CRITIC_ENABLED")
    os.environ["CRITIC_ENABLED"] = "0"
    try:
        out = qa_agent.verify({"answer": "resposta", "context": [], "verdicts": []})  # must not load the NLI model
        assert out["verdicts"] == [] and "Critic desativado" in out["answer"]
        at = _run({"DATABASE_URL": "postgresql://x", "GROQ_API_KEY": "x"}, ["q"])
        assert "Critic: desativado nesta instância" in " ".join(m.value for m in at.markdown)
    finally:
        os.environ.pop("CRITIC_ENABLED") if old is None else os.environ.__setitem__("CRITIC_ENABLED", old)
    assert qa_agent.critic_enabled()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
