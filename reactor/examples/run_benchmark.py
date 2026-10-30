"""A minimal driver: every question in questions.jsonl through client.py, one Reactor history per dataset history.

    python examples/run_benchmark.py --questions questions.jsonl --histories histories.jsonl \\
        --base-url https://tNN.<domain> --agent-id agt_... --out results.jsonl
    python predictions.py results.jsonl --histories histories.jsonl --out predictions.jsonl

REACTOR_ACCESS_KEY must be set (QUICKSTART.md). Questions are grouped by their dataset ``history_id``: for each
group, one ``--new-history``, then each of its questions in that Reactor history, one at a time. Each client line
is appended to ``--out`` as UTF-8 at once, so an interrupted run loses nothing; run it again and the questions
already in ``--out`` are skipped (delete their lines to ask again). A rerun starts a new Reactor history for each
dataset history with questions left: the agent no longer has what it saw for that history's earlier questions.
Harmless here, as every question carries its own evidence; if you load whole histories, load again first.

``retrieve()`` below is a deliberately naive keyword baseline so that the pipeline runs end to end. **Replace it
with your retrieval.** It returns excerpts in the client's evidence format: ``id`` = the session_id (the
evaluator's citations), ``date`` = that session's date in this history, ``text`` = the turns, both roles.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
BUDGET = 12000  # characters of excerpts per question: under the client's 16,000 with the instructions


def retrieve(question: dict, history: dict, k: int = 4) -> list[dict]:
    """The ``k`` turns sharing the most words with the question, each with its session's id and date."""

    words = set(re.findall(r"\w{4,}", question["question"].lower()))
    scored = []
    for session in history["sessions"]:
        for turn in session["turns"]:
            overlap = len(words & set(re.findall(r"\w{4,}", turn["content"].lower())))
            if overlap:
                scored.append((overlap, session, turn))
    scored.sort(key=lambda s: -s[0])
    return [{"id": session["session_id"], "date": session["date"],
             "text": f"{turn['role']}: {turn['content']}"[: BUDGET // k]} for _, session, turn in scored[:k]]


def client(args: argparse.Namespace, *extra: str) -> dict:
    proc = subprocess.run([sys.executable, str(args.client), "--base-url", args.base_url, "--agent-id", args.agent_id,
                           *extra], capture_output=True, text=True, encoding="utf-8")
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {"status": "error", "error": (proc.stderr or proc.stdout).strip()[-500:]}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="questions.jsonl -> client.py -> results.jsonl")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results.jsonl, appended to")
    parser.add_argument("--client", type=Path, default=KIT / "client.py")
    args = parser.parse_args(argv)

    histories = {h["history_id"]: h for h in jsonl(args.histories)}
    done = {r.get("question_id") for r in jsonl(args.out)} if args.out.exists() else set()
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in jsonl(args.questions):
        if q["question_id"] not in done:
            groups[q["history_id"]].append(q)
    print(f"{sum(map(len, groups.values()))} question(s) in {len(groups)} dataset histories to ask", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp, args.out.open("a", encoding="utf-8") as out:
        for dataset_history, questions in groups.items():
            new = client(args, "--new-history")  # one Reactor history per dataset history
            if new.get("status") != "history":
                print(f"--new-history failed: {new.get('error') or new}", file=sys.stderr)
                return 2
            for q in questions:
                evidence = Path(tmp) / "evidence.json"
                evidence.write_text(json.dumps(retrieve(q, histories[dataset_history])), encoding="utf-8")
                extra = ["--history", new["history_id"], "--question", q["question"],
                         "--question-date", q["question_date"], "--question-id", q["question_id"]]
                if json.loads(evidence.read_text(encoding="utf-8")):
                    extra += ["--evidence", str(evidence)]
                result = client(args, *extra)
                result.setdefault("question_id", q["question_id"])
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(f"{q['question_id']}: {result.get('status')} {result.get('answer', '')!s:.60}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
