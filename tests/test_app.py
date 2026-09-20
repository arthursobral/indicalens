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


def test_history_is_redrawn_cleanly_on_later_turns():
    # regression: a bare `_render(..) if .. else st.markdown(..)` was echoed by Streamlit's "magic" on turn 2+,
    # dumping a DeltaGenerator repr (and its docstring) into the page
    at = _run({"DATABASE_URL": "postgresql://x", "GROQ_API_KEY": "x"}, ["primeira", "segunda"])
    assert not at.exception
    assert not at.get("help"), "an object was echoed into the page"
    texts = " ".join(m.value for m in at.markdown)
    assert "DeltaGenerator" not in texts and "primeira" in texts and "resposta para: segunda" in texts
    assert texts.index("resposta para: segunda") < texts.index("resposta para: primeira")  # newest conversation first
    assert len(at.chat_input) == 1


def test_zero_claims_is_not_shown_as_all_clear():
    old = batch.ask_once
    batch.ask_once = lambda q, *a, **k: {**_fake(q), "claims": 0, "flagged": 0, "answer": "Não há dados suficientes."}
    os.environ.update({"DATABASE_URL": "postgresql://x", "GROQ_API_KEY": "x"})
    try:
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.chat_input[0].set_value("sem dados").run()
    finally:
        batch.ask_once = old
    html = " ".join(m.value for m in at.markdown)
    assert "sem afirmações para verificar" in html and "chip-ok" not in html.split("Busca + LLM + Critic", 1)[1]


LEAK = "OperationalError: connection to db.abcdefghij.supabase.co port 5432 failed: password authentication failed for user postgres.abcdefghij (password=hunter2pass)"


def test_errors_never_leak_hosts_or_credentials_to_visitors():
    old = batch.ask_once
    batch.ask_once = lambda q, *a, **k: {"question": q, "error": LEAK, "seconds": 0.1}
    os.environ.update({"DATABASE_URL": "postgresql://x", "GROQ_API_KEY": "x"})
    try:
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.chat_input[0].set_value("qualquer").run()
    finally:
        batch.ask_once = old
    page = " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning])
    assert any("Não consegui responder agora" in e.value for e in at.error)
    for secret in ("supabase", "hunter2pass", "abcdefghij", "OperationalError"):
        assert secret not in page, f"leaked {secret!r} to the page"


def test_report_errors_never_leak_either():
    old = batch.build_report
    batch.build_report = lambda name, qs, *a, **k: {"indicator": name, "items": [{"question": qs[0], "error": LEAK, "seconds": 0.1}],
                                                     "seconds": 0.1, "llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 1, "flagged": 0, "claims": 0}
    os.environ.update({"DATABASE_URL": "postgresql://x", "GROQ_API_KEY": "x"})
    try:
        at = AppTest.from_file(APP, default_timeout=30).run()
        next(b for b in at.button if b.key == "gen").click().run()
    finally:
        batch.build_report = old
    page = " ".join(m.value for m in at.markdown)
    assert "detalhes no log do servidor" in page
    for secret in ("supabase", "hunter2pass", "abcdefghij", "OperationalError"):
        assert secret not in page, f"leaked {secret!r} in the report"


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
