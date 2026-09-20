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
    counts = {"lookup_cases": len(lookups), "refusal_cases": len(cases.REFUSAL_CASES)}
    bcb_metrics, bcb_counts, bcb_failures = run_bcb()
    results.update(bcb_metrics)
    counts.update(bcb_counts)
    failures += bcb_failures
    return {
        "metrics": results,
        "counts": counts,
        "per_series": {k: _rate(sum(v), len(v)) for k, v in by_series.items()},
        "failures": failures,
    }


def _oracle_correlation(x_raw: dict, ipca: dict, selic: bool) -> dict:
    """Independent reimplementation: integer month index + numpy, no shared helpers with
    src/correlation.py. Same definitions (variations, 2000+, lags 0-12), so it checks the
    implementation (alignment, lag direction, CI), not the methodology."""
    import numpy as np

    ym = lambda p: int(p[:4]) * 12 + int(p[4:])
    ks = sorted(x_raw)
    dx = {}
    for a, b in zip(ks, ks[1:]):
        if ym(b) - ym(a) == 1:
            dx[ym(b)] = x_raw[b] - x_raw[a] if selic else 100 * (x_raw[b] / x_raw[a] - 1)
    iy = {ym(p): v for p, v in ipca.items() if p >= "200001"}
    best = None
    for k in range(13):
        ts = [t for t in sorted(iy) if t - k in dx]
        r = float(np.corrcoef([dx[t - k] for t in ts], [iy[t] for t in ts])[0, 1])
        if k == 0:
            r0, n0 = r, len(ts)
        if best is None or abs(r) > abs(best[1]):
            best = (k, r, len(ts))
    return {"lag": best[0], "r": best[1], "n": best[2], "r0": r0, "n0": n0}


def _revision_effect_metrics(failures: list) -> dict:
    import bisect
    import random
    from datetime import timedelta

    import numpy as np

    from src import revision_effects as fx
    from src.revisions import load_vintages, same_age_revisions

    m = {}
    vint = load_vintages()
    k1 = {r["period"]: r["revision"] for r in same_age_revisions(vint, 1)}
    k4 = {r["period"]: r["revision"] for r in same_age_revisions(vint, 4)}
    # facts read by hand from the booklets: 2023.II 3.4 (own release) -> 3.5 (next); 2022.III 3.6 -> 4.3 (4 releases later)
    facts = [k1.get("202302") == 0.1, k4.get("202203") == 0.7, len(vint) == 26, len(k1) == 25, len(k4) == 22,
             sum(v != 0 for v in k1.values()) == 5, sum(v > 0 for v in k4.values()) == 18]
    m["vintage_fact_accuracy"] = _rate(sum(facts), len(facts))
    if not all(facts):
        failures.append({"question": "vintage facts", "expected": "all 7 true", "got": str(facts)})

    evs = fx.events(vint)
    selic, usd = fx.load_market(evs)
    today = fx.date(2026, 9, 20)  # fixed so the harness does not drift with the calendar
    got = fx.analyze(evs, selic, usd, today)

    # independent oracle: numpy + bisect + own average ranks; same definitions, different code
    def step(series, d):
        days = [x[0] for x in series]
        return series[bisect.bisect_right(days, d) - 1][1]

    def ranks(a):
        a = np.asarray(a, float)
        order = a.argsort()
        rk = np.empty(len(a))
        rk[order] = np.arange(len(a))
        for v in np.unique(a):
            rk[a == v] = rk[a == v].mean()
        return rk

    ok = 0
    for name, ser, fn, after in (("selic_depois", selic, lambda a, b: a - b, True), ("selic_antes", selic, lambda a, b: a - b, False),
                                 ("dolar_depois", usd, lambda a, b: 100 * (a / b - 1), True), ("dolar_antes", usd, lambda a, b: 100 * (a / b - 1), False)):
        x, y = [], []
        for e in evs:
            if after and e["date"] + timedelta(days=fx.H) > today:
                continue
            d0, d1 = (e["date"], e["date"] + timedelta(days=fx.H)) if after else (e["date"] - timedelta(days=fx.H), e["date"])
            x.append(e["news"])
            y.append(fn(step(ser, d1), step(ser, d0)))
        r = float(np.corrcoef(x, y)[0, 1])
        rho = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
        g = got[name]
        good = g["n"] == len(x) and abs(g["r"] - r) < 1e-9 and abs(g["rho"] - rho) < 1e-9
        ok += good
        if not good:
            failures.append({"question": f"revision effect {name}", "expected": f"n={len(x)} r={r:.6f} rho={rho:.6f}", "got": str({k: g[k] for k in ("n", "r", "rho")})})
    m["revision_effect_oracle_match"] = _rate(ok, 4)

    # planted effect must be found, pure noise must not (same 25-event, mostly-zero-news shape as the real data)
    shape = [e["news"] for e in evs]
    hit = fp = 0
    seeds = range(40)
    for s in seeds:
        rnd = random.Random(s)
        planted = [(x, 1.5 * x + rnd.gauss(0, 0.5)) for x in shape]
        null = [(x, rnd.gauss(0, 1)) for x in shape]
        hit += fx.stats(planted, perms=500, seed=s)["p_perm"] < 0.05
        fp += fx.stats(null, perms=500, seed=s)["p_perm"] < 0.05
    m["revision_effect_power"] = _rate(hit, len(seeds))
    m["revision_effect_null_specificity"] = _rate(len(seeds) - fp, len(seeds))
    return m


def run_bcb() -> tuple[dict, dict, list]:
    import random

    from src import bcb_agent, bcb_client, correlation
    from src.ibge_client import fetch_series as ibge

    metrics, failures = {}, []

    # 1. lookup against hard-coded golden facts
    ok = 0
    for q, needles in cases.BCB_GOLDEN:
        got = bcb_agent.answer(q)
        good = got is not None and all(n in got["answer"] for n in needles)
        ok += good
        if not good:
            failures.append({"question": q, "expected": " & ".join(needles), "got": got and got["answer"]})
    metrics["bcb_lookup_accuracy"] = _rate(ok, len(cases.BCB_GOLDEN))

    # 2. impossible/ambiguous periods must be refused explicitly, not answered for another period
    ok = 0
    for q, needle in cases.BCB_REFUSALS:
        got = bcb_agent.answer(q)
        good = got is not None and needle in got["answer"].lower()
        ok += good
        if not good:
            failures.append({"question": q, "expected": f"message containing {needle!r}", "got": got and got["answer"]})
    metrics["bcb_refusal_accuracy"] = _rate(ok, len(cases.BCB_REFUSALS))

    # 3. correlation numbers vs the independent numpy oracle, on the real (recorded) series
    ipca = {p: float(v) for p, v in ibge(1737, 63, "all")}
    ok = 0
    pairs = [("SELIC_MENSAL", True), ("CAMBIO_USD_MEDIA_MENSAL", False)]
    for name, selic in pairs:
        raw = dict(bcb_client.fetch_series(bcb_client.SERIES[name]["code"]))
        want = _oracle_correlation(raw, ipca, selic)
        got = correlation.answer("A Selic afeta o IPCA?" if selic else "O cambio afeta o IPCA?")["result"]
        b, l0 = got["best"], got["lag0"]
        good = (b["lag"] == want["lag"] and b["n"] == want["n"] and abs(b["r"] - want["r"]) < 1e-9
                and l0["n"] == want["n0"] and abs(l0["r"] - want["r0"]) < 1e-9)
        ok += good
        if not good:
            failures.append({"question": name, "expected": str(want), "got": str({"lag": b["lag"], "r": b["r"], "n": b["n"]})})
    metrics["correlation_oracle_match"] = _rate(ok, len(pairs))

    # 4. planted structure: a known lag must be recovered; pure noise and independent random
    # walks (in variations) must NOT be reported as related
    ms = [correlation.add_months("200001", i) for i in range(200)]
    hit = noise_ok = walk_ok = 0
    seeds = range(40)
    for s in seeds:
        rnd = random.Random(s)
        x = {m: rnd.gauss(0, 1) for m in ms}
        y = {m: x.get(correlation.add_months(m, -3), 0) + rnd.gauss(0, 0.5) for m in ms}
        hit += correlation.analyze(x, y)["best"]["lag"] == 3
        n1 = {m: rnd.gauss(0, 1) for m in ms}
        n2 = {m: rnd.gauss(0, 1) for m in ms}
        noise_ok += not correlation.analyze(n1, n2)["best"]["distinguishable"]
        a = b2 = 0.0
        w1, w2 = {}, {}
        for m in ms:
            a += rnd.gauss(0, 1)
            b2 += rnd.gauss(0, 1)
            w1[m], w2[m] = a, b2
        walk_ok += not correlation.analyze(correlation._diff(w1), correlation._diff(w2))["best"]["distinguishable"]
    metrics["planted_lag_recall"] = _rate(hit, len(seeds))
    metrics["noise_specificity"] = _rate(noise_ok, len(seeds))
    metrics["random_walk_specificity"] = _rate(walk_ok, len(seeds))

    # 5. recognition of correlation questions in varied phrasing (dev vs held-out)
    for label, cs in (("dev", cases.CORR_ROUTE_DEV), ("heldout", cases.CORR_ROUTE_TEST)):
        pos = [(q, correlation.detect(q) is not None) for q, want in cs if want]
        neg = [(q, correlation.detect(q) is None) for q, want in cs if not want]
        metrics[f"corr_route_recall_{label}"] = _rate(sum(g for _, g in pos), len(pos))
        metrics[f"corr_route_specificity_{label}"] = _rate(sum(g for _, g in neg), len(neg))
        failures += [{"question": q, "expected": "recognized as correlation", "got": "not recognized"} for q, g in pos if not g]
        failures += [{"question": q, "expected": "not correlation", "got": "recognized as correlation"} for q, g in neg if not g]

    # 6. GDP revisions vs Selic/dollar: hand-verified vintage facts, independent numpy oracle, planted effect/null
    metrics.update(_revision_effect_metrics(failures))

    counts = {"bcb_golden": len(cases.BCB_GOLDEN), "bcb_refusals": len(cases.BCB_REFUSALS), "planted_seeds": len(seeds),
              "corr_route_dev": len(cases.CORR_ROUTE_DEV), "corr_route_heldout": len(cases.CORR_ROUTE_TEST)}
    return metrics, counts, failures


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
    _install_bcb(mode)


def _install_bcb(mode: str) -> None:
    """Same idea for the Banco Central client: recorded JSON keyed by code + dates. fetch_latest
    is left alone (it depends on today's date, so it has hermetic unit tests instead)."""
    import functools
    import json
    from datetime import date

    import requests

    import src.bcb_client as bc

    orig_series, orig_range = bc.fetch_series, bc.fetch_range

    def _load_or_fetch(key, call):
        path = FIXTURES / (re.sub(r"[^\w.-]", "_", "bcb_" + key) + ".json")
        if mode == "fixtures":
            if not path.exists():
                raise FileNotFoundError(f"no fixture {path.name}; run `python -m eval.run --record`")
            return json.loads(path.read_text(encoding="utf-8"))
        for attempt in range(1, 5):
            try:
                data = call()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if mode == "record":
            FIXTURES.mkdir(exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        return data

    @functools.lru_cache(maxsize=None)
    def fetch_series(code, start="01/01/1995", end=None):
        # default end is "today": the key must not carry it, or the fixture name would drift daily
        key = f"series_{code}_{start}" if end is None else f"series_{code}_{start}_{end}"
        return [tuple(p) for p in _load_or_fetch(key, lambda: orig_series(code, start, end))]

    @functools.lru_cache(maxsize=None)
    def fetch_range(code, start, end):
        key = f"range_{code}_{start:%Y%m%d}_{end:%Y%m%d}"
        def call():
            try:
                return [[d.isoformat(), v] for d, v in orig_range(code, start, end)]
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    return []  # SGS answers 404 for a window before the series exists: record it as "no data"
                raise

        data = _load_or_fetch(key, call)
        return [(date.fromisoformat(d), v) for d, v in data]

    bc.fetch_series, bc.fetch_range = fetch_series, fetch_range


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
