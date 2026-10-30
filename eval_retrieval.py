"""Offline retrieval metrics for retrieval.py on a dev split: no Reactor calls.

    python eval_retrieval.py dev [--k 1 3 5 10] [--cite 2] [--limit N] [--misses FILE]

For answerable questions, per question_type and overall:
  hit@k    any gold session in the top k sessions
  all@k    every gold session in the top k
  sent     share of gold sessions among the excerpts evidence() would send
  F1@cite  evidence F1 if the top --cite sessions were cited
Then the top-1 session score, answerable vs unanswerable: a first look at an abstention threshold.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from retrieval import HistoryIndex

HISTORY_ID = re.compile(r'\s*\{\s*"history_id"\s*:\s*"([^"]+)"')


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stream_histories(path: Path, wanted: set[str]):
    """Yield the wanted histories, parsing only their lines (dev-large is 2.4 GB)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = HISTORY_ID.match(line)
            if m and m.group(1) not in wanted:
                continue
            if not line.strip():
                continue
            history = json.loads(line)
            if history["history_id"] in wanted:
                yield history


def f1(cited: list[str], gold: set[str]) -> float:
    hits = len(set(cited) & gold)
    if not hits:
        return 0.0
    p, r = hits / len(set(cited)), hits / len(gold)
    return 2 * p * r / (p + r)


def quantiles(values: list[float]) -> str:
    if not values:
        return "n=0"
    v = sorted(values)
    pick = lambda q: v[min(int(q * len(v)), len(v) - 1)]
    return f"n={len(v):<4} median={statistics.median(v):6.2f}  p10={pick(0.1):6.2f}  p90={pick(0.9):6.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", type=Path, help="directory with histories.jsonl, questions.jsonl, gold.jsonl")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--cite", type=int, default=2, help="sessions cited for F1@cite")
    ap.add_argument("--limit", type=int, help="an evenly spread sample of N questions")
    ap.add_argument("--misses", type=Path, help="write answerable questions with no gold session in the top 5")
    args = ap.parse_args()

    questions = read_jsonl(args.split / "questions.jsonl")
    gold = {g["question_id"]: g for g in read_jsonl(args.split / "gold.jsonl")}
    if args.limit and args.limit < len(questions):
        n = len(questions)
        questions = [questions[i * n // args.limit] for i in range(args.limit)]
    by_history: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_history[q["history_id"]].append(q)

    depth = max(max(args.k), args.cite, 5)
    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    top1 = {False: [], True: []}
    misses = []
    started, done = time.time(), 0
    # one HistoryIndex per history_id, built when its line streams past and dropped after its questions
    for history in stream_histories(args.split / "histories.jsonl", set(by_history)):
        index = HistoryIndex(history)
        for q in by_history.pop(history["history_id"]):
            g = gold[q["question_id"]]
            ranked = index.search(q["question"], k_sessions=depth)
            top1[g["abstain"]].append(ranked[0][1] if ranked else 0.0)
            if g["abstain"]:
                continue
            ids = [sid for sid, _ in ranked]
            ev = set(g["evidence"])
            sent = {e["id"] for e in index.evidence(q["question"])}
            m = {}
            for k in args.k:
                m[f"hit@{k}"] = float(bool(ev & set(ids[:k])))
                m[f"all@{k}"] = float(ev <= set(ids[:k]))
            m["sent"] = len(ev & sent) / len(ev)
            m[f"F1@{args.cite}"] = f1(ids[:args.cite], ev)
            for key in (g["question_type"], "all"):
                for name, value in m.items():
                    rows[key][name].append(value)
            if not ev & set(ids[:5]):
                misses.append({**q, "question_type": g["question_type"], "answer": g["answer"],
                               "evidence": g["evidence"], "top5": ranked[:5]})
        done += 1
        if done % 50 == 0:
            print(f"  {done} histories, {time.time() - started:.0f}s", file=sys.stderr)
    if by_history:
        print(f"warning: {len(by_history)} histories not found in histories.jsonl", file=sys.stderr)

    names = [f"{p}@{k}" for p in ("hit", "all") for k in args.k] + ["sent", f"F1@{args.cite}"]
    order = sorted(k for k in rows if k != "all") + ["all"]
    width = max(len(k) for k in order)
    print(f"\n{'type':<{width}}  {'n':>4}  " + "  ".join(f"{n:>7}" for n in names))
    for key in order:
        r = rows[key]
        print(f"{key:<{width}}  {len(r['sent']):>4}  " + "  ".join(f"{sum(r[n]) / len(r[n]):7.3f}" for n in names))
    print(f"\ntop-1 session score  answerable    {quantiles(top1[False])}")
    print(f"                     unanswerable  {quantiles(top1[True])}")
    print(f"\n{done} histories in {time.time() - started:.0f}s")

    if args.misses:
        with args.misses.open("w", encoding="utf-8") as f:
            for miss in misses:
                f.write(json.dumps(miss, ensure_ascii=False) + "\n")
        print(f"{len(misses)} miss(es) written to {args.misses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
