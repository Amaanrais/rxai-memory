"""A stand-in for reactor/client.py that calls nothing: it appends what it was asked (its arguments and the evidence
file's bytes) to $FAKE_CLIENT_LOG and prints a result line. A question id in $FAKE_TIMEOUT_ONCE (comma-separated)
times out the first time it is asked."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def option(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def main(argv: list[str]) -> int:
    log = Path(os.environ["FAKE_CLIENT_LOG"])
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    evidence = option(argv, "--evidence")
    call = {"argv": argv, "evidence": Path(evidence).read_text(encoding="utf-8") if evidence else None}
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(call) + "\n")
    if "--new-history" in argv:
        print(json.dumps({"status": "history", "history_id": f"rh_{len(calls):04d}"}))
        return 0
    qid = option(argv, "--question-id")
    ref = f"{len(calls):08x}"
    times_out = qid in os.environ.get("FAKE_TIMEOUT_ONCE", "").split(",")
    if times_out and not any(option(c["argv"], "--question-id") == qid for c in calls):
        print(json.dumps({"status": "timeout", "question_id": qid, "history_id": option(argv, "--history"),
                          "ref": ref, "late_replies": []}))
        return 0
    print(json.dumps({"status": "answered", "question_id": qid, "history_id": option(argv, "--history"), "ref": ref,
                      "answer": "fake [E1]", "reply": f"# FINAL ANSWER [ref:{ref}]: fake [E1]", "late_replies": [],
                      "citations": [], "abstained": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
