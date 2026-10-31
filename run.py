"""The benchmark driver: every question through reactor/client.py with our retrieval, one Reactor history per dataset
history. Based on reactor/examples/run_benchmark.py.

    python run.py --questions dev/questions.jsonl --histories dev/histories.jsonl \\
        --base-url https://tNN.<domain> --agent-id agt_... --out results.jsonl
    python make_predictions.py results.jsonl --questions dev/questions.jsonl --out predictions.jsonl

REACTOR_ACCESS_KEY must be set (reactor/QUICKSTART.md). Histories are streamed: only the ones the questions need are
parsed, and each gets one HistoryIndex and one --new-history, reused for all its questions. Each client line is
appended to --out at once; a rerun skips the questions already answered there and asks the others again (timeouts,
refusals, errors). The excerpts sent and suggest_citations() for each question go to --evidence-log (default:
<out>.evidence.jsonl), which make_predictions.py reads.

Timeouts: the client waits --timeout seconds (default 600). A timeout doesn't cancel the question on Reactor's side,
so the driver first sends nothing and listens on the same Reactor history for up to --late-wait seconds (default 120;
listen.py). Only if no answer arrives is the question asked once more in the same history. A late answer that
arrives in another question's late_replies is recorded as an answered line marked "recovered"
(make_predictions.recover_late()).

Resuming: a rerun carries on in the Reactor history of the latest result line (the agent's current one) without a
reset, and finishes that dataset history first; if the agent's history has changed meanwhile, it starts afresh.

Startup: a new history restarts the agent, and a question sent before it is back up fails at once ("closed", HTTP
403 or "history_resetting"). So after --new-history the driver waits --settle seconds (default 15; the kit has no
readiness call), and a question that fails that way before anything is answered in its Reactor history is asked
again in the same history after 10, 20 and 40 seconds (STARTUP_RETRIES). Every attempt is recorded;
make_predictions.choose() keeps the answered line.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from eval_retrieval import read_jsonl, stream_histories
from listen import wait_for_late
from make_predictions import recover_late
from retrieval import HistoryIndex, is_aggregate

ROOT = Path(__file__).resolve().parent
MAX_CHARS = 16000  # the client's default
ROUTED_MAX_CHARS = 30000  # routed questions send 24,000 characters of evidence; the server takes 32,000

INSTRUCTIONS = ("Put the [E#] labels of the passages you used on the FINAL ANSWER line itself, e.g. 'June [E2]'. "
                "Answer with the single value only, no hedging.")
UPDATE_INSTRUCTION = "If a value changed, the most recent date wins."
STARTUP_RETRIES = (10, 20, 40)  # seconds before each new attempt at a question the restarting agent dropped
UPDATE = re.compile(r"\b(current|currently|now|latest|most recent|most recently|still|anymore|these days|so far"
                    r"|at the moment|nowadays|recently|new|changed|switched|updated"
                    r"|have i|do i|am i|how often|since)\b")
# asking for the old value: "most recent wins" would be wrong
PAST = re.compile(r"\b(previous|previously|initially|originally|first|former|earlier|used to|before i)\b")


def is_update(question: str) -> bool:
    """A question about a value that may have changed, so its latest mention wins (dev: 54 of 78
    knowledge-update questions, and 81 of the other 448, mostly present-tense ones)."""
    q = question.lower()
    return bool(UPDATE.search(q)) and not PAST.search(q)


def prompt(question: str) -> str:
    """The question as sent: the user's words, then our instructions."""
    parts = [question.strip(), INSTRUCTIONS]
    if is_update(question):
        parts.append(UPDATE_INSTRUCTION)
    return " ".join(parts)


def client(args: argparse.Namespace, *extra: str) -> dict:
    proc = subprocess.run([sys.executable, str(args.client), "--base-url", args.base_url, "--agent-id", args.agent_id,
                           *extra], capture_output=True, text=True, encoding="utf-8")
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {"status": "error", "error": (proc.stderr or proc.stdout).strip()[-500:]}


def pause(seconds: float) -> None:
    time.sleep(seconds)


def startup_failure(result: dict) -> bool:
    """A failure that means the agent wasn't up yet: the chat closed, the connection refused (HTTP 403, "the agent
    isn't running"), or the new history still starting."""
    return (result.get("status") == "closed"
            or (result.get("status") == "error" and "HTTP 403" in str(result.get("error")))
            or (result.get("status") == "refused" and result.get("code") == "history_resetting"))


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def resume_point(lines: list[dict], dataset_history: dict[str, str]) -> tuple[str, str] | None:
    """(dataset history, Reactor history) to carry on in without a reset: the dataset history of the latest result
    line the client wrote, if it has an answered question, with that line's Reactor history. Only the most
    recently active one: the agent's current history is the last one used."""
    direct = [r for r in lines if not r.get("recovered") and r.get("history_id")
              and r.get("question_id") in dataset_history]
    if not direct:
        return None
    last = direct[-1]
    current = dataset_history[last["question_id"]]
    if any(r.get("status") == "answered" and dataset_history.get(r.get("question_id")) == current for r in lines):
        return current, last["history_id"]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="questions.jsonl -> client.py -> results.jsonl")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results.jsonl, appended to")
    parser.add_argument("--evidence-log", type=Path, help="default: <out>.evidence.jsonl, appended to")
    parser.add_argument("--timeout", default="600", help="passed to client.py (seconds; default 600)")
    parser.add_argument("--late-wait", type=float, default=120.0,
                        help="seconds to listen for a timed-out question's late answer before asking it again, "
                             "once (default 120)")
    parser.add_argument("--settle", type=float, default=15.0,
                        help="seconds to wait after --new-history for the agent to come up (default 15)")
    parser.add_argument("--client", type=Path, default=ROOT / "reactor" / "client.py")
    args = parser.parse_args(argv)
    evidence_log = args.evidence_log or args.out.with_suffix(".evidence.jsonl")
    if not os.environ.get("REACTOR_ACCESS_KEY"):
        print("set REACTOR_ACCESS_KEY first (reactor/QUICKSTART.md, step 2)", file=sys.stderr)
        return 2

    all_questions = read_jsonl(args.questions)
    dataset_history = {q["question_id"]: q["history_id"] for q in all_questions}
    lines = read_jsonl(args.out) if args.out.exists() else []
    late = recover_late(lines)  # answers to earlier timeouts that arrived late, not yet recorded
    answered = {r.get("question_id") for r in lines + late if r.get("status") == "answered"}
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in all_questions:
        if q["question_id"] not in answered:
            groups[q["history_id"]].append(q)
    total = sum(map(len, groups.values()))
    resume = resume_point(lines, dataset_history)
    resume = resume if resume and resume[0] in groups else None
    print(f"{total} question(s) in {len(groups)} dataset histories to ask "
          f"({len(answered)} already answered in {args.out})", file=sys.stderr)
    if resume:
        print(f"resuming dataset history {resume[0]} in Reactor history {resume[1]} (no reset)", file=sys.stderr)

    timeout = ["--timeout", args.timeout]
    asked, spent = 0, 0.0
    with tempfile.TemporaryDirectory() as tmp, args.out.open("a", encoding="utf-8") as out, \
            evidence_log.open("a", encoding="utf-8") as log:

        def record(result: dict) -> None:
            """Append a result line, then any late answers it carries for earlier timed-out questions."""
            out.write(json.dumps(result) + "\n")
            out.flush()
            lines.append(result)
            for line in recover_late(lines):
                out.write(json.dumps(line) + "\n")
                out.flush()
                lines.append(line)
                print(f"  recovered {line['question_id']}'s late answer (ref {line['ref']}): "
                      f"{line['answer']!s:.60}", file=sys.stderr)

        def record_late(question_id: str, late: list[dict]) -> None:
            """Record the answers to earlier timed-out questions among final answers heard while listening."""
            probe = {"question_id": question_id, "late_replies": late}  # not written: only what it recovers
            for line in recover_late(lines + [probe]):
                line["recovered"]["via"] = "listening after the timeout"
                out.write(json.dumps(line) + "\n")
                out.flush()
                lines.append(line)
                print(f"  recovered {line['question_id']}'s late answer (ref {line['ref']}): "
                      f"{line['answer']!s:.60}", file=sys.stderr)

        for line in late:
            record(line)
        if not total:
            return 0
        # the resumed dataset history first: any --new-history before it would end its Reactor history
        first = stream_histories(args.histories, {resume[0]}) if resume else iter(())
        rest = stream_histories(args.histories, set(groups) - ({resume[0]} if resume else set()))
        for history in itertools.chain(first, rest):
            questions = groups.pop(history["history_id"])
            index = HistoryIndex(history)
            if resume and history["history_id"] == resume[0]:
                reactor_history = resume[1]
            else:
                reactor_history = None
            for q in questions:
                evidence = index.evidence(q["question"], question_date=q["question_date"])
                suggested = index.suggest_citations(q["question"])
                log.write(json.dumps({"question_id": q["question_id"], "history_id": q["history_id"],
                                      "evidence": evidence, "suggested": suggested}, ensure_ascii=False) + "\n")
                log.flush()
                path = Path(tmp) / "evidence.json"
                path.write_text(json.dumps(evidence), encoding="utf-8")
                max_chars = ROUTED_MAX_CHARS if is_aggregate(q["question"]) else MAX_CHARS
                started = time.monotonic()
                retried = fresh = False
                startup = iter(STARTUP_RETRIES)
                while True:
                    if reactor_history is None:
                        new = client(args, "--new-history", *timeout)  # one Reactor history per dataset history
                        if new.get("status") != "history":
                            print(f"--new-history failed: {new.get('error') or new}", file=sys.stderr)
                            return 2
                        reactor_history = new["history_id"]
                        print(f"  new Reactor history {reactor_history} (running: {new.get('running')}, "
                              f"restarted: {new.get('restarted')}); waiting {args.settle:g}s for the agent",
                              file=sys.stderr)
                        if args.settle > 0:
                            pause(args.settle)
                    extra = ["--history", reactor_history, "--question", prompt(q["question"]),
                             "--question-date", q["question_date"], "--question-id", q["question_id"],
                             "--max-chars", str(max_chars), *timeout]
                    if evidence:
                        extra += ["--evidence", str(path)]
                    result = client(args, *extra)
                    result.setdefault("question_id", q["question_id"])
                    if (result.get("status") == "refused" and not fresh
                            and result.get("code") in ("history_changed", "history_required")):
                        # the resumed history is no longer the agent's current one: start this one afresh
                        print(f"  {q['question_id']}: {result['code']}; starting a new Reactor history",
                              file=sys.stderr)
                        reactor_history, fresh = None, True
                        continue
                    record(result)
                    if startup_failure(result) and not any(
                            r.get("status") == "answered" and r.get("history_id") == reactor_history for r in lines):
                        wait = next(startup, None)
                        if wait is not None:
                            print(f"  {q['question_id']}: {result.get('status')} "
                                  f"({result.get('code') or result.get('error') or 'chat closed'}) before the agent "
                                  f"was up; asking again in the same Reactor history in {wait}s", file=sys.stderr)
                            pause(wait)
                            continue
                    if result.get("status") == "timeout" and not retried:
                        retried = True
                        print(f"  {q['question_id']}: timeout after {time.monotonic() - started:.0f}s; listening up "
                              f"to {args.late_wait:g}s for its late answer (ref {result.get('ref')})", file=sys.stderr)
                        heard = wait_for_late(args.base_url, args.agent_id, reactor_history,
                                              [result["ref"]] if result.get("ref") else [], args.late_wait)
                        record_late(q["question_id"], heard)
                        answer = next((line for line in reversed(lines) if line.get("status") == "answered"
                                       and line.get("question_id") == q["question_id"]), None)
                        if answer:
                            result = answer
                            break
                        print(f"  {q['question_id']}: no late answer; asking again in the same Reactor history",
                              file=sys.stderr)
                        continue
                    break
                took = time.monotonic() - started
                asked += 1
                spent += took
                average = spent / asked
                cited = len(result.get("citations") or [])
                print(f"[{asked}/{total}] {q['question_id']}: {result.get('status')} {took:5.1f}s "
                      f"(avg {average:.1f}s, ETA {clock(average * (total - asked))}) cited={cited} "
                      f"{result.get('answer', result.get('error', ''))!s:.60}", file=sys.stderr)
    if groups:
        print(f"warning: {len(groups)} dataset histories not found in {args.histories}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
