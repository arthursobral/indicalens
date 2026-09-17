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

Semana 2 de 12 concluida: alem da ingestao tabular da semana 1 (IPCA, PIB,
desocupacao), agora tambem ingerimos texto corrido de verdade — os
comentarios analiticos que o IBGE publica a cada trimestre sobre a PNAD
Continua — para que o agente responda perguntas mais descritivas, nao so
"qual foi o numero". Ver [docs/03-roadmap.md](docs/03-roadmap.md) para o
plano completo (pasta local, nao versionada — ver secao Docs abaixo).

## Arquitetura

```
IBGE SIDRA API ------------> src/ingest.py --\
                                               +--> embeddings locais (MiniLM) --> pgvector (Supabase)
IBGE (comentarios em PDF) -> src/notes.py ----/                                        |
                                                                                        |
pergunta do usuario --> src/qa_agent.py (LangGraph: retrieve -> generate) <-------------'
                                              |
                                        Groq / Ollama (LLM de sintese)
```

- `src/ingest.py`: dados tabulares (IPCA, PIB, desocupacao, rendimento,
  informalidade) viram uma frase citavel por ponto de dado.
- `src/notes.py`: baixa o caderno trimestral "Indicadores IBGE" (PDF), extrai
  a secao de Comentarios e quebra em trechos citaveis por tema (ex: "Taxa de
  Desocupacao", "Populacao Ocupada").

O agente de QA (`src/qa_agent.py`) e um grafo LangGraph de 2 nos:

- **retrieve**: embeda a pergunta e busca os chunks mais proximos por
  similaridade de cosseno no Postgres/pgvector.
- **generate**: manda os chunks recuperados (cada um com sua citacao) pro LLM
  e exige que a resposta cite as fontes numeradas.

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
4. Ingira os indicadores (ultimos 24 periodos de cada serie por padrao) e o
   texto dos comentarios do trimestre mais recente:
   ```bash
   python -m src.ingest
   python -m src.notes
   ```
5. Pergunte:
   ```bash
   python -m src.qa_agent "Qual foi a variacao mensal do IPCA no ultimo mes disponivel?"
   ```

## Testes

```bash
python tests/test_ibge_client.py
python tests/test_notes.py
```