# IndicaLens

Sistema multi-agente que le indicadores publicos do IBGE (e, nas proximas
semanas, do Banco Central) e responde perguntas sobre eles com citacao de
fonte obrigatoria — nada de numero solto sem apontar de onde veio.

O projeto existe para provar, com codigo rodando, o que normalmente fica so
no papel num portfolio de IA: RAG com grounding verificavel, orquestracao
multi-agente com LangGraph, harness de avaliacao com gate de CI, e um agente
de verificacao que rejeita respostas sem suporte nos dados recuperados.

Dados 100% publicos e sem chave de API: [IBGE/SIDRA](https://servicodados.ibge.gov.br/api/docs/agregados)
para indicadores (IPCA, PIB, PNAD Continua) e [Banco Central (SGS)](https://api.bcb.gov.br/dados/serie)
para Selic e cambio.

## Status

Semanas 9-10 de 12 concluidas; semana 11: interface Streamlit pronta (`app.py`), deploy pendente; antes disso, semanas 3-8: alem da ingestao tabular e do texto corrido das semanas
1-2, agora tem um Table Agent (lookup direto e deterministico pra perguntas
numericas) e um Indicator Analyst Agent (classifica cada comentario por tema
via zero-shot). Ver [docs/03-roadmap.md](docs/03-roadmap.md) para o plano
completo (pasta local, nao versionada — ver secao Docs abaixo).

## Arquitetura

```
IBGE SIDRA API ------------> src/ingest.py --\
                                               +--> embeddings locais (MiniLM) --> pgvector (Supabase)
IBGE (comentarios em PDF) -> src/notes.py ----/         |
                                                         v
                                          src/indicator_analyst.py (zero-shot: tema de cada comentario)

pergunta do usuario --> src/qa_agent.py (LangGraph)
                            |
                            +--> src/table_agent.py (lookup direto, sem LLM) --> resposta com citacao
                            |
                            +--> retrieve (pgvector) --> generate (Groq/Ollama) --> resposta com citacao
```

- `src/ingest.py`: dados tabulares (IPCA, PIB, desocupacao, rendimento,
  informalidade, taxa de pobreza pelas linhas internacional e nacional)
  viram uma frase citavel por ponto de dado.
- `src/notes.py`: baixa o caderno trimestral "Indicadores IBGE" (PDF), extrai
  a secao de Comentarios e quebra em trechos citaveis por tema (ex: "Taxa de
  Desocupacao", "Populacao Ocupada").
- `src/indicator_analyst.py`: classifica cada chunk de comentario por tema
  (emprego, inflacao, PIB, renda, pobreza, informalidade) via zero-shot.
- `src/revisions.py`: quanto o IBGE revisou o PIB trimestral desde a primeira
  divulgacao (cadernos do FTP vs. valor atual da API).
- `src/critic.py`: depois do RAG, checa cada afirmacao da resposta contra os
  trechos recuperados (NLI local) e sinaliza o que nao tem suporte.
- `src/revision_effects.py`: as revisoes do PIB antecedem movimentos da Selic ou do
  dolar? Revisao com mesma idade (cadernos do IBGE em `data/pib_vintages.json`),
  teste de permutacao e controle "antes"; resultado atual: sem evidencia.
  Ex: `python -m src.qa_agent "As revisoes do PIB afetam a Selic?"`.
- `src/batch.py`: gera um relatorio por indicador (perguntas fixas pelo grafo
  inteiro) com tempo, tokens e alertas do Critic por relatorio.
  `python -m src.batch [--only IPCA,PIB]` -> `reports/*.md` + `summary.json`.
- `src/bcb_agent.py`: Selic (meta) e dolar direto do Banco Central (mais recente, data,
  mes ou ano), sem LLM. Ex: `python -m src.qa_agent "qual a taxa selic mais recente?"`.
- `src/correlation.py`: Correlation Agent. Relaciona Selic/cambio com o IPCA
  (variacoes, defasagens 0-12, IC ajustado por multiplas defasagens); sem LLM.
  Ex: `python -m src.qa_agent "A inflacao se move junto com a Selic?"`.
- `src/bcb_client.py`: series mensais do Banco Central (SGS): Selic e cambio
  (dolar), sem chave. Base do Correlation Agent (em construcao, semanas 9-10).
- `src/table_agent.py`: perguntas numericas sobre uma serie conhecida (ex:
  "qual foi a taxa de desocupacao mais recente?") sao respondidas por lookup
  direto na API do IBGE, sem passar pelo LLM.

O agente de QA (`src/qa_agent.py`) e um grafo LangGraph com roteamento: tenta
o Table Agent primeiro; se a pergunta nao citar uma serie conhecida, cai pro
RAG vetorial de sempre (retrieve -> generate).

## Seguranca

Segredos (`DATABASE_URL`, `GROQ_API_KEY`, chaves do Langfuse) ficam so no `.env` local ou nos
Secrets do Streamlit Cloud; `.env*`, `secrets.toml`, chaves/certificados, `docs/` e `reports/`
estao no `.gitignore` (so os modelos `.env.example` e `.streamlit/secrets.toml.example`, com
placeholders, sao versionados). `tests/test_no_secrets.py` roda no CI e reprova se algum arquivo
rastreado tiver formato de chave (Groq, Langfuse, URL de banco com senha, JWT, chave privada,
token do GitHub), se um arquivo secreto for rastreado ou se as regras de ignore forem
enfraquecidas; o GitHub secret scanning e o push protection tambem estao ativos. O app nunca
mostra a um visitante o texto bruto de um erro (pode conter host ou usuario do banco): ele vai
so para o log do servidor.

## Demo online

**https://indicalens.streamlit.app/** (Streamlit Community Cloud, plano gratuito). Verificado no
site publico em 20/09/2026: consultas diretas (IPCA mais recente, 1,6 s) e correlacao (Cambio x
IPCA, r = 0,26, 3,7 s) funcionam. **Perguntas que usam o LLM e o Critic (comentarios do IBGE) funcionam, mas sao muito lentas
na nuvem gratuita**: a 1a levou 294 s (baixar e carregar os modelos, ~2 min, mais a inferencia) e a
2a, ja com os modelos carregados, passou de 100 s sem terminar (localmente: 10-30 s). O gargalo e
a CPU da hospedagem gratuita (o Critic roda um modelo NLI local); os logs nao mostram erro nem
falta de memoria. Em investigacao. O app dorme apos alguns dias sem visitas; a
primeira visita o acorda (~30 s). Limite de 15 perguntas por sessao para proteger a cota gratuita.

## Interface web

```bash
.venv/Scripts/python -m streamlit run app.py
```

Visual proprio (tema escuro, marca em SVG, etiquetas por caminho e status do Critic; funciona no celular). Abas: **Perguntas** (chat com o mesmo grafo da CLI, mostrando caminho, tempo, tokens,
alertas do Critic e fontes), **Relatorios** (batch por indicador, com download do `.md`) e
**Sobre** (limites honestos). Variaveis opcionais: `MAX_QUESTIONS_PER_SESSION` (padrao 25) e
`CRITIC_ENABLED=0` para hospedar com pouca memoria (o modelo do Critic usa ~1 GB). Passos de
deploy no Streamlit Community Cloud e riscos: ver `.streamlit/secrets.toml.example` e a pasta
local `docs/`. **Deploy ainda nao feito.**

## Setup

1. Crie um projeto gratuito no [Supabase](https://supabase.com), habilite a
   extensao `pgvector` e rode [scripts/schema.sql](scripts/schema.sql) no SQL
   editor.
2. Copie `.env.example` para `.env` e preencha `DATABASE_URL` (Supabase >
   Project Settings > Database > Connection string). Para o LLM, use
   `GROQ_API_KEY` (gratuito em [console.groq.com](https://console.groq.com))
   ou deixe em branco e rode um [Ollama](https://ollama.com) local.
3. Instale as dependencias:
   ```bash
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt   # Windows
   ```
4. Ingira os indicadores (todo o historico disponivel de cada serie por
   padrao — desde 1979 pro IPCA, por exemplo) e o texto dos comentarios do
   trimestre mais recente:
   ```bash
   python -m src.ingest
   python -m src.notes
   python -m src.indicator_analyst
   python -m src.revisions
   ```
5. (Opcional) Tracing com [Langfuse](https://cloud.langfuse.com): preencha
   `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` no `.env` para ver, por pergunta,
   cada passo do grafo, latencia e tokens. Sem as chaves, nada e enviado.
6. Pergunte:
   ```bash
   python -m src.qa_agent "Qual foi a variacao mensal do IPCA no ultimo mes disponivel?"
   python -m src.qa_agent "Qual foi a taxa de desocupacao mais recente?"
   ```

## Avaliacao

```bash
python -m eval.run --tier offline --gate   # roda no CI, sem segredos (IBGE + BCB + correlacao)
python -m eval.run --tier full             # local: Groq + Supabase + Critic
```

## Testes

```bash
python tests/test_ibge_client.py
python tests/test_notes.py
python tests/test_table_agent.py
python tests/test_indicator_analyst.py
python tests/test_critic.py
python tests/test_revisions.py
```