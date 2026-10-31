"""run.py with the fake client (tests/fake_client.py): it waits after --new-history, asks a question the restarting
agent dropped again in the same Reactor history, and a rerun asks only what isn't answered yet. Nothing calls Reactor.

    python -m unittest discover tests
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_client.py"
DEV = ROOT / "dev"
BASE = ["--base-url", "https://fake.invalid", "--agent-id", "agt_fake", "--client", str(FAKE)]


def first_questions(n: int) -> list[dict]:
    return [json.loads(line) for line in (DEV / "questions.jsonl").read_text(encoding="utf-8").splitlines()[:n]]


@unittest.skipUnless((DEV / "histories.jsonl").exists(), "needs dev/ (not in the repository)")
class StartupFailures(unittest.TestCase):
    def drive(self, questions: list[dict], env: dict, existing: list[dict] = ()):
        """run.main() in-process: (exit code, pauses, client calls' arguments, result lines)."""
        pauses: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            qpath, out, log = Path(tmp) / "q.jsonl", Path(tmp) / "r.jsonl", Path(tmp) / "calls.jsonl"
            qpath.write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
            if existing:
                out.write_text("".join(json.dumps(r) + "\n" for r in existing), encoding="utf-8")
            env = {"REACTOR_ACCESS_KEY": "fake", "FAKE_CLIENT_LOG": str(log), **env}
            with mock.patch.dict(os.environ, env), mock.patch.object(run, "pause", pauses.append), \
                    mock.patch.object(run, "wait_for_late", lambda *a: []):
                code = run.main(["--questions", str(qpath), "--histories", str(DEV / "histories.jsonl"), *BASE,
                                 "--out", str(out)])
            calls = [json.loads(line)["argv"] for line in log.read_text(encoding="utf-8").splitlines()] \
                if log.exists() else []
            lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        return code, pauses, calls, lines

    def test_waits_then_retries_in_the_same_history(self):
        questions = first_questions(3)
        code, pauses, calls, lines = self.drive(questions, {"FAKE_STARTUP_FAILS": "closed,403,resetting"})
        self.assertEqual(code, 0)
        self.assertEqual(sum("--new-history" in c for c in calls), len({q["history_id"] for q in questions}))
        first = questions[0]["question_id"]
        asks = [c for c in calls if "--question-id" in c and c[c.index("--question-id") + 1] == first]
        self.assertEqual(len(asks), 4)
        self.assertEqual(len({c[c.index("--history") + 1] for c in asks}), 1)
        self.assertEqual(pauses[:4], [15.0, 10, 20, 40])
        self.assertEqual([r["status"] for r in lines[:4]], ["closed", "error", "refused", "answered"])
        self.assertEqual({r["question_id"] for r in lines if r["status"] == "answered"},
                         {q["question_id"] for q in questions})

    def test_gives_up_after_three_retries(self):
        code, pauses, _, lines = self.drive(first_questions(1), {"FAKE_STARTUP_FAILS": "closed,closed,closed,closed"})
        self.assertEqual(code, 0)
        self.assertEqual(pauses, [15.0, 10, 20, 40])
        self.assertEqual([r["status"] for r in lines], ["closed"] * 4)

    def test_rerun_asks_only_failed_and_missing_questions(self):
        questions = first_questions(3)
        # an earlier run: the first question answered, the second closed, the third never asked
        existing = [{"status": "answered", "question_id": questions[0]["question_id"], "history_id": "rh_old",
                     "ref": "00000001", "answer": "x [E1]", "late_replies": []},
                    {"status": "closed", "question_id": questions[1]["question_id"], "history_id": "rh_old2",
                     "ref": "00000002", "late_replies": []}]
        code, _, calls, lines = self.drive(questions, {}, existing)
        self.assertEqual(code, 0)
        asked = [c[c.index("--question-id") + 1] for c in calls if "--question-id" in c]
        self.assertEqual(asked, [q["question_id"] for q in questions[1:]])
        self.assertEqual(lines[:2], existing)  # appended to, never rewritten


if __name__ == "__main__":
    unittest.main()
