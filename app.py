"""IndicaLens web UI (Streamlit). Run locally: streamlit run app.py

Questions go through the same LangGraph as the CLI (src.batch.ask_once adds retry and timing).
Secrets: on Streamlit Community Cloud they come from st.secrets and are copied to the environment
BEFORE src is imported (src.config reads the environment); locally they come from .env.
"""

import os

import streamlit as st

try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # no secrets file locally: .env is used instead
    pass

st.set_page_config(page_title="IndicaLens", page_icon="📊", layout="wide")

from src import batch, llm, qa_agent  # noqa: E402  (after the secrets are in the environment)

MAX_QUESTIONS = int(os.environ.get("MAX_QUESTIONS_PER_SESSION", "25"))  # protects the free LLM quota of a public demo
EXAMPLES = [
    "Qual foi o IPCA mais recente?",
    "Qual foi a taxa de desocupacao em maio de 2025?",
    "Qual a taxa selic mais recente?",
    "A inflacao se move junto com a Selic?",
    "O cambio afeta o IPCA?",
    "O PIB do 2 trimestre de 2022 foi revisado?",
    "As revisoes do PIB afetam a Selic ou o dolar?",
    "O que os comentarios do IBGE dizem sobre a desocupacao?",
]
PATH_LABEL = {"deterministic": "consulta direta (sem LLM)", "rag": "busca + LLM + Critic"}


@st.cache_resource(show_spinner="Iniciando...")
def _warm() -> bool:
    from src import qa_agent  # noqa: F401  (imports the graph; the local models load lazily on the first RAG question, which is why that one is slower)

    return True


def _missing_config() -> list[str]:
    need = ["DATABASE_URL"]
    if not (os.environ.get("GROQ_API_KEY") or os.environ.get("OLLAMA_HOST")):
        need.append("GROQ_API_KEY")
    return [k for k in need if not os.environ.get(k)]


def _ask(question: str) -> dict:
    before = dict(llm.USAGE)
    item = batch.ask_once(question)
    item["tokens_in"] = llm.USAGE["input"] - before["input"]
    item["tokens_out"] = llm.USAGE["output"] - before["output"]
    return item


def _render(item: dict) -> None:
    if "error" in item:
        st.error(f"Não consegui responder agora ({item['error']}). Tente de novo em instantes.")
        return
    st.markdown(item["answer"].replace("\n", "\n\n"))  # single newlines would glue paragraphs onto the previous bullet
    parts = [f"caminho: {PATH_LABEL[item['path']]}", f"tempo: {item['seconds']} s"]
    if item["path"] == "rag":
        parts += [f"tokens: {item['tokens_in']} entrada / {item['tokens_out']} saída",
                  f"Critic: {item['flagged']} de {item['claims']} afirmações sinalizadas" if qa_agent.critic_enabled()
                  else "Critic: desativado nesta instância"]
    st.caption(" · ".join(parts))
    with st.expander("Fontes"):
        for s in item["sources"] or ["nenhuma recuperada"]:
            st.markdown(f"- {s}")


def chat_tab() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages, st.session_state.asked = [], 0
    with st.sidebar:
        st.subheader("Exemplos")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            _render(m["item"]) if m["role"] == "assistant" else st.markdown(m["text"])
    question = st.chat_input("Pergunte sobre IPCA, PIB, PNAD, Selic ou câmbio") or st.session_state.pop("pending", None)
    if not question:
        return
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if st.session_state.asked >= MAX_QUESTIONS:
            st.warning(f"Limite de {MAX_QUESTIONS} perguntas por sessão nesta demo (protege a cota gratuita do LLM). Recarregue a página para recomeçar.")
            return
        st.session_state.asked += 1
        with st.spinner("Consultando as fontes..."):
            item = _ask(question)
        _render(item)
    st.session_state.messages.append({"role": "assistant", "item": item})


def reports_tab() -> None:
    st.write("Um relatório por indicador: perguntas fixas passando pelo grafo inteiro, com tempo, tokens e alertas do Critic.")
    name = st.selectbox("Indicador", list(batch.REPORTS))
    if st.button("Gerar relatório"):
        with st.spinner("Gerando..."):
            r = batch.build_report(name, batch.REPORTS[name])
        c = st.columns(4)
        c[0].metric("Tempo", f"{r['seconds']} s")
        c[1].metric("Chamadas ao LLM", r["llm_calls"])
        c[2].metric("Tokens (ent./saída)", f"{r['tokens_in']} / {r['tokens_out']}")
        c[3].metric("Critic", f"{r['flagged']} de {r['claims']}")
        md = batch.to_markdown(r)
        st.markdown(md.replace("\n", "\n\n"))
        st.download_button("Baixar .md", md, file_name=f"{name}.md")


def about_tab() -> None:
    st.markdown(
        """
**IndicaLens** responde perguntas sobre indicadores públicos do IBGE (IPCA, PIB, PNAD) e do Banco Central (Selic, câmbio),
sempre citando a fonte. Números pedidos diretamente vêm de consulta por código, sem LLM; comentários do IBGE passam por
busca + LLM, e um **Critic** confere cada afirmação contra os trechos recuperados.

**Como ler as respostas (limites honestos)**
- O Critic sinaliza cerca de 1 em cada 3 afirmações; nas afirmações rotuladas à mão, ~87% a 94% estavam sustentadas pelas fontes.
- Correlação não prova causa. Selic × IPCA: r = 0,13, não distinguível de zero; dólar × IPCA: r = 0,26 com 4 meses de defasagem, fraca.
- Revisões do PIB × Selic/dólar: sem evidência, mas o teste tem pouco poder (24 divulgações, 6 com revisão).
- Não é recomendação de investimento.

Código, testes e documentação: veja o repositório no GitHub (arthursobral/indicalens).
"""
    )


st.title("📊 IndicaLens")
st.caption("Indicadores do IBGE e do Banco Central, com fonte e verificação.")
if missing := _missing_config():
    st.error("Configuração ausente: " + ", ".join(missing) + ". Defina no .env (local) ou nos Secrets do Streamlit Cloud.")
    st.stop()
_warm()
tab_chat, tab_reports, tab_about = st.tabs(["Perguntas", "Relatórios", "Sobre"])
with tab_chat:
    chat_tab()
with tab_reports:
    reports_tab()
with tab_about:
    about_tab()
