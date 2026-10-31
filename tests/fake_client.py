"""A stand-in for reactor/client.py that calls nothing: it appends what it was asked (its arguments and the evidence
file's bytes) to $FAKE_CLIENT_LOG and prints a result line. A question id in $FAKE_TIMEOUT_ONCE (comma-separated)
times out the first time it is asked. $FAKE_STARTUP_FAILS (comma-separated: closed, 403, resetting) is how the first
asks in each new Reactor history fail, in order, as an agent that is still restarting would."""
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
    history = option(argv, "--history")
    earlier = sum(option(c["argv"], "--history") == history for c in calls)  # asks in this history so far
    fails = [f for f in os.environ.get("FAKE_STARTUP_FAILS", "").split(",") if f]
    if earlier < len(fails):
        fail = fails[earlier]
        base = {"question_id": qid, "history_id": history, "ref": ref}
        print(json.dumps({**base, "status": "closed", "other_replies": [], "late_replies": [], "retries": 0}
                         if fail == "closed" else
                         {**base, "status": "error", "error": "refused at connect (HTTP 403): wrong key, no access "
                                                              "to this agent, or the agent isn't running"}
                         if fail == "403" else
                         {**base, "status": "refused", "code": "history_resetting", "forwarded": False}))
        return 5 if fail == "closed" else 2 if fail == "403" else 4
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
