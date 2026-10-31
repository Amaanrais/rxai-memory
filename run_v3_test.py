"""The test-run driver for version 3: run.py's driver with run_v3.py's evidence and question text.

    python run_v3_test.py --questions test/questions.jsonl --histories test/histories.jsonl \\
        --base-url https://tNN.<domain> --agent-id agt_... --out results-test-v3.jsonl
    python make_predictions.py results-test-v3.jsonl --questions test/questions.jsonl --out predictions.jsonl

From run.py (imported, not copied where it can be): one Reactor history per dataset history, a rerun resumes the
latest dataset history in its Reactor history without a reset, --timeout 600, and a timed-out question is listened
for (--late-wait, listen.py) before it is asked once more in the same Reactor history; late answers that arrive in
other questions' late_replies are recorded as "recovered" lines (make_predictions.recover_late()).
From run_v3.py (run_v2.py with version 3): evidence(version=3), prompt_v2() and the evidence log's fields. Its two
flags are on by default here, as the test run uses them: no missing-words hint (--hint puts it back) and the
lenient abstain line (--strict leaves it out). tests/test_run_v3_test.py proves, with a fake client, that the
question text and evidence sent are byte-identical to run_v3.py --no-hint --lenient.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from eval_retrieval import read_jsonl, stream_histories
from listen import wait_for_late
from make_predictions import recover_late
from retrieval import HistoryIndex, route
from run import MAX_CHARS, ROOT, ROUTED_MAX_CHARS, client, clock, resume_point
from run_v2 import HINT_WORDS, prompt_v2

VERSION = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="questions.jsonl -> client.py -> results.jsonl (version 3, test run)")
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
    parser.add_argument("--client", type=Path, default=ROOT / "reactor" / "client.py")
    parser.add_argument("--hint", action="store_true", help="add the missing-words hint (run_v3.py's default)")
    parser.add_argument("--strict", action="store_true", help="leave out the lenient abstain line")
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
          f"({len(answered)} already answered in {args.out}); hint {'on' if args.hint else 'off'}, "
          f"{'strict' if args.strict else 'lenient'}", file=sys.stderr)
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
                evidence = index.evidence(q["question"], question_date=q["question_date"], version=VERSION)
                suggested = index.suggest_citations(q["question"])
                missing = index.missing_words(q["question"], evidence, limit=HINT_WORDS)
                log.write(json.dumps({"question_id": q["question_id"], "history_id": q["history_id"],
                                      "version": VERSION, "evidence": evidence, "suggested": suggested,
                                      "missing": missing, "hint": args.hint, "lenient": not args.strict},
                                     ensure_ascii=False) + "\n")
                log.flush()
                path = Path(tmp) / "evidence.json"
                path.write_text(json.dumps(evidence), encoding="utf-8")
                question = prompt_v2(q["question"], missing if args.hint else [], not args.strict)
                max_chars = ROUTED_MAX_CHARS if route(q["question"], VERSION)[1] > 12000 else MAX_CHARS
                started = time.monotonic()
                retried = fresh = False
                while True:
                    if reactor_history is None:
                        new = client(args, "--new-history", *timeout)  # one Reactor history per dataset history
                        if new.get("status") != "history":
                            print(f"--new-history failed: {new.get('error') or new}", file=sys.stderr)
                            return 2
                        reactor_history = new["history_id"]
                    extra = ["--history", reactor_history, "--question", question,
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
