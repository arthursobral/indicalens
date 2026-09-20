"""LLM call for answer synthesis: Groq REST API if GROQ_API_KEY is set,
otherwise a local Ollama server. Both are OpenAI/chat-style JSON APIs, so a
plain `requests.post` avoids pulling in either vendor's SDK.
"""

import requests

from src.config import GROQ_API_KEY, GROQ_MODEL, OLLAMA_HOST, OLLAMA_MODEL
from src.tracing import observe, record_generation


# running totals of LLM usage in this process; the batch job reads deltas per report
USAGE = {"calls": 0, "input": 0, "output": 0}


def _count(input_tokens: int, output_tokens: int) -> None:
    USAGE["calls"] += 1
    USAGE["input"] += input_tokens
    USAGE["output"] += output_tokens


@observe(as_type="generation")
def chat(system: str, user: str) -> str:
    if GROQ_API_KEY:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage", {})
        _count(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        record_generation(model=GROQ_MODEL, usage_details={"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)})
        return body["choices"][0]["message"]["content"]

    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    _count(body.get("prompt_eval_count", 0), body.get("eval_count", 0))
    record_generation(model=OLLAMA_MODEL, usage_details={"input": body.get("prompt_eval_count", 0), "output": body.get("eval_count", 0)})
    return body["message"]["content"]
