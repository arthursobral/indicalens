"""IndicaLens web UI (Streamlit). Run locally: streamlit run app.py

Questions go through the same LangGraph as the CLI (src.batch.ask_once adds retry and timing).
Secrets: on Streamlit Community Cloud they come from st.secrets and are copied to the environment
BEFORE src is imported (src.config reads the environment); locally they come from .env.
Look and feel: .streamlit/config.toml (theme) + assets/style.css + assets/logo.svg.
"""

import os
from pathlib import Path

import streamlit as st

try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # no secrets file locally: .env is used instead
    pass

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="IndicaLens", page_icon=str(ASSETS / "favicon.png"), layout="wide")

from src import batch, llm, qa_agent  # noqa: E402  (after the secrets are in the environment)

MAX_QUESTIONS = int(os.environ.get("MAX_QUESTIONS_PER_SESSION", "25"))  # protects the free LLM quota of a public demo
EXAMPLES = [  # (tag, question)
    ("Consulta direta", "Qual foi o IPCA mais recente?"),
    ("Consulta direta", "Qual foi a taxa de desocupacao em maio de 2025?"),
    ("Banco Central", "Qual a taxa selic mais recente?"),
    ("Correlação", "A inflacao se move junto com a Selic?"),
    ("Correlação", "O cambio afeta o IPCA?"),
    ("Revisões do PIB", "O PIB do 2 trimestre de 2022 foi revisado?"),
    ("Revisões do PIB", "As revisoes do PIB afetam a Selic ou o dolar?"),
    ("Comentários do IBGE (usa LLM)", "O que os comentarios do IBGE dizem sobre a desocupacao?"),
]
PATH_CHIP = {"deterministic": ("Consulta direta · sem LLM", "direct"), "rag": ("Busca + LLM + Critic", "rag")}


def _css() -> None:
    st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def chip(text: str, kind: str = "muted") -> str:
    return f'<span class="chip chip-{kind}">{text}</span>'


def _msg(role: str):
    return st.chat_message(role, avatar=str(ASSETS / "favicon.png") if role == "assistant" else ":material/person:")


def _hero() -> None:
    logo = (ASSETS / "logo.svg").read_text(encoding="utf-8")
    st.markdown(
        f"""<div class="hero">
  <div class="brand">{logo}<div><h1>IndicaLens</h1>
  <p>Indicadores do IBGE e do Banco Central, com a fonte em cada número.</p></div></div>
  <div class="chips">{chip("IPCA · PIB · PNAD", "direct")}{chip("Selic · câmbio", "direct")}{chip("Correlações com intervalo de confiança", "rag")}{chip("Cada afirmação verificada pelo Critic", "rag")}</div>
</div>""",
        unsafe_allow_html=True,
    )


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


def _result_chips(item: dict) -> str:
    label, kind = PATH_CHIP[item["path"]]
    chips = [chip(label, kind), chip(f"{item['seconds']} s")]
    if item["path"] == "rag":
        chips.append(chip(f"{item['tokens_in']} → {item['tokens_out']} tokens"))
        if not qa_agent.critic_enabled():
            chips.append(chip("Critic: desativado nesta instância", "warn"))
        elif not item["claims"]:  # nothing was checked: must not look like "checked and fine"
            chips.append(chip("Critic: sem afirmações para verificar"))
        elif item["flagged"]:
            chips.append(chip(f"Critic: {item['flagged']} de {item['claims']} afirmações sinalizadas", "warn"))
        else:
            chips.append(chip(f"Critic: 0 de {item['claims']} afirmações sinalizadas", "ok"))
    return f'<div class="chips result-chips">{"".join(chips)}</div>'


def _render(item: dict) -> None:
    if "error" in item:
        st.error(f"Não consegui responder agora ({item['error']}). Tente de novo em instantes.")
        return
    st.markdown(item["answer"].replace("\n", "\n\n"))  # single newlines would glue paragraphs onto the previous bullet
    st.markdown(_result_chips(item), unsafe_allow_html=True)
    with st.expander("Fontes"):
        for s in item["sources"] or ["nenhuma recuperada"]:
            st.markdown(f"- {s}")


def _sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="side-title">Como funciona</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="side-step"><b>1</b><span>Números pedidos direto vêm da API, por código, sem LLM.</span></div>
<div class="side-step"><b>2</b><span>Comentários do IBGE passam por busca + LLM.</span></div>
<div class="side-step"><b>3</b><span>O Critic confere cada afirmação contra as fontes e sinaliza o que não se sustenta.</span></div>""",
            unsafe_allow_html=True,
        )
        used = st.session_state.get("asked", 0)
        st.markdown(
            f'<div class="side-note">Perguntas nesta sessão: <b>{used} de {MAX_QUESTIONS}</b> (limite da demo, para proteger a cota gratuita do LLM).<br>'
            "Correlação não prova causa. Não é recomendação de investimento.</div>",
            unsafe_allow_html=True,
        )


def _examples_grid() -> None:
    st.markdown('<div class="empty-title">Comece por uma pergunta</div>'
                '<div class="empty-sub">Clique em um exemplo ou escreva a sua abaixo.</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (tag, q) in enumerate(EXAMPLES):
        with cols[i % 2]:
            st.markdown(f'<div class="tag">{tag}</div>', unsafe_allow_html=True)
            if st.button(q, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()


def chat_tab() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages, st.session_state.asked = [], 0
    # input on top, conversations newest-first: the newest answer appears right under the input
    question = st.chat_input("Pergunte sobre IPCA, PIB, PNAD, Selic ou câmbio") or st.session_state.pop("pending", None)
    if not st.session_state.messages and not question:
        _examples_grid()
        return
    earlier = list(zip(st.session_state.messages[::2], st.session_state.messages[1::2]))  # (user, assistant) pairs
    if question:
        with _msg("user"):
            st.markdown(question)
        with _msg("assistant"):
            if st.session_state.asked >= MAX_QUESTIONS:
                st.warning(f"Limite de {MAX_QUESTIONS} perguntas por sessão nesta demo (protege a cota gratuita do LLM). Recarregue a página para recomeçar.")
            else:
                st.session_state.asked += 1
                with st.spinner("Consultando as fontes..."):
                    item = _ask(question)
                _render(item)
                st.session_state.messages += [{"role": "user", "text": question}, {"role": "assistant", "item": item}]
    for user, assistant in reversed(earlier):
        with _msg("user"):
            st.markdown(user["text"])
        with _msg("assistant"):
            _render(assistant["item"])


def reports_tab() -> None:
    st.markdown("Um relatório por indicador: perguntas fixas passando pelo grafo inteiro, com tempo, tokens e alertas do Critic.")
    name = st.selectbox("Indicador", list(batch.REPORTS))
    if st.button("Gerar relatório", key="gen"):
        with st.spinner("Gerando..."):
            r = batch.build_report(name, batch.REPORTS[name])
        c = st.columns(4)
        c[0].metric("Tempo", f"{r['seconds']} s")
        c[1].metric("Chamadas ao LLM", r["llm_calls"])
        c[2].metric("Tokens (ent. / saída)", f"{r['tokens_in']} / {r['tokens_out']}")
        c[3].metric("Critic", f"{r['flagged']} de {r['claims']}")
        md = batch.to_markdown(r)
        st.markdown(md.replace("\n", "\n\n"))
        st.download_button("Baixar .md", md, file_name=f"{name}.md")


def about_tab() -> None:
    a, b = st.columns(2)
    a.markdown(
        """<div class="card"><h4>O que é</h4>
<p>Responde perguntas sobre indicadores públicos do IBGE (IPCA, PIB, PNAD) e do Banco Central (Selic, câmbio), sempre citando a fonte.
Números pedidos diretamente vêm de consulta por código, sem LLM. Comentários do IBGE passam por busca + LLM, e um <b>Critic</b> confere
cada afirmação contra os trechos recuperados.</p></div>""",
        unsafe_allow_html=True,
    )
    b.markdown(
        """<div class="card"><h4>Como ler as respostas</h4><ul>
<li>O Critic sinaliza cerca de 1 em cada 3 afirmações; nas rotuladas à mão, 87% a 94% estavam sustentadas pelas fontes.</li>
<li>Correlação não prova causa.</li><li>Não é recomendação de investimento.</li></ul></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, stat, title, text in (
        (c1, "r = 0,13", "Selic × IPCA", "Fraca e não distinguível de zero (320 meses, jan/2000 em diante)."),
        (c2, "r = 0,26", "Dólar × IPCA", "Fraca, com defasagem de 4 meses; intervalo ajustado por múltiplas defasagens."),
        (c3, "sem evidência", "Revisões do PIB × Selic/dólar", "Teste de baixo poder: 24 divulgações, só 6 com revisão."),
    ):
        col.markdown(f'<div class="card short"><div class="stat">{stat}</div><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="foot">Código, testes e documentação: <a href="https://github.com/arthursobral/indicalens" target="_blank" rel="noopener">github.com/arthursobral/indicalens</a></div>',
        unsafe_allow_html=True,
    )


_css()
_hero()
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
_sidebar()
