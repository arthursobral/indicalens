"""Eval harness runner.

    python -m eval.run --tier offline --gate   # CI: recorded fixtures, no network, no secrets
    python -m eval.run --tier offline --live   # against the real IBGE API
    python -m eval.run --tier offline --record # refresh eval/fixtures from the real API
    python -m eval.run --tier full             # local: whole graph + Critic evaluation

Prints metrics, writes eval/last_report.json (gitignored), and with --gate
exits 1 if any metric is below its floor in eval/thresholds.json.
"""

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

from eval import cases

HERE = Path(__file__).parent


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def _has_value(answer: str, expected: str) -> bool:
    """Expected value as a standalone number ('0.3%' must not match '-0.3%' or '10.3%')."""
    return re.search(r"(?<![\d.\-])" + re.escape(expected), answer) is not None


def _rate(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


# --- offline tier -------------------------------------------------------------

def run_offline() -> dict:
    from src.table_agent import answer

    results, failures = {}, []
    by_series: dict[str, list[int]] = {}
    ok_lookup = 0
    lookups = cases.lookup_cases()
    for c in lookups:
        got = answer(c["question"])
        ok = got is not None and _has_value(got["answer"], c["expected_value"]) and c["expected_period"] in got["answer"].lower()
        ok_lookup += ok
        by_series.setdefault(c["series"], []).append(int(ok))
        if not ok:
            failures.append({"question": c["question"], "expected": f"{c['expected_value']} / {c['expected_period']}", "got": got and got["answer"]})

    ok_refuse = 0
    for q in cases.REFUSAL_CASES:
        got = answer(q)
        ok_refuse += got is None
        if got is not None:
            failures.append({"question": q, "expected": "None (decline)", "got": got["answer"]})

    results["lookup_accuracy"] = _rate(ok_lookup, len(lookups))
    results["refusal_accuracy"] = _rate(ok_refuse, len(cases.REFUSAL_CASES))
    return {
        "metrics": results,
        "counts": {"lookup_cases": len(lookups), "refusal_cases": len(cases.REFUSAL_CASES)},
        "per_series": {k: _rate(sum(v), len(v)) for k, v in by_series.items()},
        "failures": failures,
    }


# --- full tier ----------------------------------------------------------------

def _ask(question: str) -> tuple[dict, float]:
    """qa_agent.ask with retry on the Groq free tier's 429 rate limit."""
    import requests
    from src.qa_agent import ask

    for attempt in range(1, 4):
        start = time.time()
        try:
            return ask(question), time.time() - start
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 429 or attempt == 3:
                raise
            time.sleep(20 * attempt)


def run_full() -> dict:
    from src import critic
    from src.critic import is_refusal

    failures, latency = [], {"table": [], "revision": [], "rag": []}
    verdicts_all = []
    counts = {"route": [0, 0], "decline": [0, 0], "multi": [0, 0], "prose": [0, 0]}

    def record(kind, ok, q, expected, got):
        counts[kind][0] += bool(ok)
        counts[kind][1] += 1
        if not ok:
            failures.append({"question": q, "expected": expected, "got": got})

    for q in cases.FULL_ROUTE:
        r, t = _ask(q)
        latency["table"].append(t)
        record("route", r["context"] and r["context"][0]["score"] == 1.0, q, "deterministic path (score 1.0)", r["answer"])
    for q, must in cases.FULL_REVISION:
        r, t = _ask(q)
        latency["revision"].append(t)
        record("route", r["context"] and r["context"][0]["score"] == 1.0 and must in r["answer"], q, f"revision path with '{must}'", r["answer"])

    def rag(kind, q, check, expected):
        r, t = _ask(q)
        latency["rag"].append(t)
        verdicts_all.extend(r["verdicts"])
        via_rag = bool(r["context"]) and r["context"][0]["score"] != 1.0  # a deterministic answer here is a routing bug
        record(kind, via_rag and check(r["answer"]), q, expected + " via RAG", r["answer"])
        time.sleep(3)  # stay under the free tier's rate limit

    for q in cases.FULL_DECLINE:
        rag("decline", q, is_refusal, "admits lack of data")
    for q, years in cases.FULL_MULTI:
        rag("multi", q, lambda a, y=years: all(x in a for x in y), f"mentions {years}")
    for q, frags in cases.FULL_PROSE:
        rag("prose", q, lambda a, f=frags: any(_norm(x) in _norm(a) for x in f), f"one of {frags}")

    supported = sum(v["status"] == "supported" for v in verdicts_all)
    inferred = sum(v["status"] == "inferred" for v in verdicts_all)  # reasonable conclusions, not errors (owner policy)
    critic_stats = {"good": [0, 0], "wrong": [0, 0], "invented": [0, 0]}
    for c in cases.critic_cases():
        kind = "good" if c["expect"] == ["supported"] else "wrong" if c["expect"] == ["flagged"] else "invented"
        status = [v["status"] for v in critic.verify(c["answer"], c["ctx"])]
        ok = all(s == "supported" for s in status) and status != [] if kind == "good" else any(s in critic.HARD_FLAGS for s in status)
        critic_stats[kind][0] += ok
        critic_stats[kind][1] += 1
        if not ok:
            failures.append({"question": c["answer"], "expected": f"critic: {kind}", "got": status})

    # The Critic against labeled real claims (no LLM calls): the check the
    # planted cases above cannot give, because those are short and clean.
    labeled = json.loads((HERE / "critic_labeled.json").read_text(encoding="utf-8"))
    ok_sup = sup_total = ok_flag = flag_total = 0
    for item in labeled["items"]:
        ctx = [{"content": labeled["chunks"][i]} for i in item["chunks"]]
        flagged = critic.verify_claim(item["claim"], [], ctx)["status"] in critic.HARD_FLAGS
        if item["label"] == 1:
            sup_total += 1
            ok_sup += not flagged
        else:
            flag_total += 1
            ok_flag += flagged
            if not flagged:
                failures.append({"question": item["claim"], "expected": "flagged (label 0)", "got": "supported"})
    n_labeled = sup_total + flag_total

    def pct(xs, p):
        return round(sorted(xs)[min(len(xs) - 1, int(len(xs) * p))], 2) if xs else None

    return {
        "metrics": {
            "routing_accuracy": _rate(*counts["route"]),
            "decline_accuracy": _rate(*counts["decline"]),
            "multi_period_accuracy": _rate(*counts["multi"]),
            "prose_accuracy": _rate(*counts["prose"]),
            "faithfulness": _rate(supported + inferred, len(verdicts_all)),  # inferences are not errors
            "strict_faithfulness": _rate(supported, len(verdicts_all)),
            "inference_rate": _rate(inferred, len(verdicts_all)),
            "critic_flag_rate": _rate(len(verdicts_all) - supported - inferred, len(verdicts_all)),
            "critic_specificity": _rate(*critic_stats["good"]),
            "critic_recall_wrong_number": _rate(*critic_stats["wrong"]),
            "critic_recall_invented": _rate(*critic_stats["invented"]),
            "critic_real_specificity": _rate(ok_sup, sup_total),
            "critic_real_recall": _rate(ok_flag, flag_total),
            "labeled_faithfulness": _rate(sup_total, n_labeled),
        },
        "counts": {"claims_checked": len(verdicts_all), **{k: v[1] for k, v in counts.items()},
                   **{f"critic_{k}": v[1] for k, v in critic_stats.items()}, "labeled_claims": n_labeled},
        "latency_seconds": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95), "n": len(v)} for k, v in latency.items()},
        "failures": failures,
    }


# --- report + gate -------------------------------------------------------------

def gate(tier: str, metrics: dict) -> bool:
    floors = json.loads((HERE / "thresholds.json").read_text(encoding="utf-8"))[tier]
    ok = True
    print(f"\n-- gate ({tier}) --")
    for name, floor in floors.items():
        value = metrics.get(name)
        passed = value is not None and value >= floor
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {value} (floor {floor})")
    return ok


FIXTURES = HERE / "fixtures"


def _install_fetch(mode: str) -> None:
    """Replaces fetch_series for the harness (production code untouched).

    fixtures (default): read recorded JSON from eval/fixtures/, no network.
        The gate must be more reliable than the code it guards; CI hit
        ConnectTimeout on the live IBGE API (rate limit on the runner's IP)
        and, worse, other live tests printed SKIPPED and stayed green. The
        system under test and the oracle read the same snapshot, so what is
        exercised is still the parsing/routing logic, where the bugs lived.
    live: real API, cached per process, retry with backoff (local use).
    record: live + write the fixtures (refresh with `--record`).
    """
    import functools
    import json

    import requests

    import src.ibge_client as ic
    import src.table_agent as ta

    original = ic.fetch_series

    @functools.lru_cache(maxsize=None)
    def fetch(agregado, variavel, periodos="all", classificacao=None, nivel_territorial="N1", localidade="1"):
        key = f"{agregado}_{variavel}_{periodos}_{classificacao or 'none'}_{nivel_territorial}_{localidade}"
        path = FIXTURES / (re.sub(r"[^\w.-]", "_", key) + ".json")
        if mode == "fixtures":
            if not path.exists():
                raise FileNotFoundError(f"no fixture {path.name}; run `python -m eval.run --record`")
            return [tuple(p) for p in json.loads(path.read_text(encoding="utf-8"))]
        for attempt in range(1, 5):
            try:
                points = original(agregado, variavel, periodos, classificacao, nivel_territorial, localidade)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if mode == "record":
            FIXTURES.mkdir(exist_ok=True)
            path.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")
        return points

    ic.fetch_series = ta.fetch_series = cases.fetch_series = fetch


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["offline", "full", "all"], default="offline")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--live", action="store_true", help="hit the real IBGE API instead of eval/fixtures")
    ap.add_argument("--record", action="store_true", help="hit the real API and rewrite eval/fixtures")
    args = ap.parse_args()
    _install_fetch("record" if args.record else "live" if args.live else "fixtures")

    tiers = ["offline", "full"] if args.tier == "all" else [args.tier]
    report, healthy = {}, True
    for tier in tiers:
        start = time.time()
        report[tier] = run_offline() if tier == "offline" else run_full()
        report[tier]["duration_seconds"] = round(time.time() - start, 1)
        r = report[tier]
        print(f"\n== {tier} ({r['duration_seconds']}s) ==")
        for k, v in r["metrics"].items():
            print(f"  {k}: {v}")
        print(f"  counts: {r['counts']}")
        if "per_series" in r:
            print(f"  per_series: {r['per_series']}")
        if "latency_seconds" in r:
            print(f"  latency: {r['latency_seconds']}")
        for f in r["failures"]:
            print(f"  FAILED: {f['question']}\n      expected: {f['expected']}\n      got:      {f['got']}")
        if args.gate:
            healthy &= gate(tier, r["metrics"])
    (HERE / "last_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
