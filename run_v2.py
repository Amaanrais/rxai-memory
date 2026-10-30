"""run.py with version 2 evidence and the missing-words hint. run.py stays as it was, so v1 runs reproduce.

    python run_v2.py --questions dev/questions.jsonl --histories dev/histories.jsonl \\
        --base-url https://tNN.<domain> --agent-id agt_... --out results-v2.jsonl
    python make_predictions.py results-v2.jsonl --questions dev/questions.jsonl --out predictions-v2.jsonl

Changes from run.py:
- evidence(version=2): a hit turn too long for its room keeps the stretch with the most question words;
- the missing-words hint: the question's rarer content words that appear nowhere in the evidence (at most
  HINT_WORDS), named in the question so that the agent abstains on a near miss. The evidence log records them.
Everything else (one HistoryIndex and --new-history per dataset history, instructions, --max-chars, results
appended at once, answered questions skipped on rerun, latency/ETA) is run.py's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from eval_retrieval import read_jsonl, stream_histories
from retrieval import HistoryIndex, is_aggregate
from run import MAX_CHARS, ROOT, ROUTED_MAX_CHARS, client, clock, prompt

VERSION = 2
HINT_WORDS = 4


def hint(words: list[str]) -> str:
    return (f"Note: the evidence never mentions: {', '.join(words)}. If the answer depends on these, answer "
            "INSUFFICIENT_EVIDENCE; otherwise answer normally.")


def prompt_v2(question: str, missing: list[str]) -> str:
    """run.py's prompt, plus the hint when some question words are missing from the evidence."""
    text = prompt(question)
    return f"{text} {hint(missing)}" if missing else text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="questions.jsonl -> client.py -> results.jsonl (version 2)")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results.jsonl, appended to")
    parser.add_argument("--evidence-log", type=Path, help="default: <out>.evidence.jsonl, appended to")
    parser.add_argument("--timeout", help="passed to client.py (seconds)")
    parser.add_argument("--client", type=Path, default=ROOT / "reactor" / "client.py")
    args = parser.parse_args(argv)
    evidence_log = args.evidence_log or args.out.with_suffix(".evidence.jsonl")
    if not os.environ.get("REACTOR_ACCESS_KEY"):
        print("set REACTOR_ACCESS_KEY first (reactor/QUICKSTART.md, step 2)", file=sys.stderr)
        return 2

    answered = ({r.get("question_id") for r in read_jsonl(args.out) if r.get("status") == "answered"}
                if args.out.exists() else set())
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in read_jsonl(args.questions):
        if q["question_id"] not in answered:
            groups[q["history_id"]].append(q)
    total = sum(map(len, groups.values()))
    print(f"{total} question(s) in {len(groups)} dataset histories to ask "
          f"({len(answered)} already answered in {args.out})", file=sys.stderr)
    if not total:
        return 0

    timeout = ["--timeout", args.timeout] if args.timeout else []
    asked, spent = 0, 0.0
    with tempfile.TemporaryDirectory() as tmp, args.out.open("a", encoding="utf-8") as out, \
            evidence_log.open("a", encoding="utf-8") as log:
        for history in stream_histories(args.histories, set(groups)):
            questions = groups.pop(history["history_id"])
            index = HistoryIndex(history)
            new = client(args, "--new-history", *timeout)  # one Reactor history per dataset history
            if new.get("status") != "history":
                print(f"--new-history failed: {new.get('error') or new}", file=sys.stderr)
                return 2
            for q in questions:
                evidence = index.evidence(q["question"], question_date=q["question_date"], version=VERSION)
                suggested = index.suggest_citations(q["question"])
                missing = index.missing_words(q["question"], evidence, limit=HINT_WORDS)
                log.write(json.dumps({"question_id": q["question_id"], "history_id": q["history_id"],
                                      "version": VERSION, "evidence": evidence, "suggested": suggested,
                                      "missing": missing}, ensure_ascii=False) + "\n")
                log.flush()
                path = Path(tmp) / "evidence.json"
                path.write_text(json.dumps(evidence), encoding="utf-8")
                max_chars = ROUTED_MAX_CHARS if is_aggregate(q["question"]) else MAX_CHARS
                extra = ["--history", new["history_id"], "--question", prompt_v2(q["question"], missing),
                         "--question-date", q["question_date"], "--question-id", q["question_id"],
                         "--max-chars", str(max_chars), *timeout]
                if evidence:
                    extra += ["--evidence", str(path)]
                started = time.monotonic()
                result = client(args, *extra)
                took = time.monotonic() - started
                result.setdefault("question_id", q["question_id"])
                out.write(json.dumps(result) + "\n")
                out.flush()
                asked += 1
                spent += took
                average = spent / asked
                cited = len(result.get("citations") or [])
                print(f"[{asked}/{total}] {q['question_id']}: {result.get('status')} {took:5.1f}s "
                      f"(avg {average:.1f}s, ETA {clock(average * (total - asked))}) cited={cited} "
                      f"missing={','.join(missing) or '-'} "
                      f"{result.get('answer', result.get('error', ''))!s:.60}", file=sys.stderr)
    if groups:
        print(f"warning: {len(groups)} dataset histories not found in {args.histories}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
