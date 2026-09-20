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

Semanas 3-4 de 12: alem da ingestao tabular e do texto corrido das semanas
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
- `src/table_agent.py`: perguntas numericas sobre uma serie conhecida (ex:
  "qual foi a taxa de desocupacao mais recente?") sao respondidas por lookup
  direto na API do IBGE, sem passar pelo LLM.

O agente de QA (`src/qa_agent.py`) e um grafo LangGraph com roteamento: tenta
o Table Agent primeiro; se a pergunta nao citar uma serie conhecida, cai pro
RAG vetorial de sempre (retrieve -> generate).

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
5. Pergunte:
   ```bash
   python -m src.qa_agent "Qual foi a variacao mensal do IPCA no ultimo mes disponivel?"
   python -m src.qa_agent "Qual foi a taxa de desocupacao mais recente?"
   ```

## Avaliacao

```bash
python -m eval.run --tier offline --gate   # roda no CI, sem segredos
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