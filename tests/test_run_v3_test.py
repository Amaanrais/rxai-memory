"""run_v3_test.py sends what run_v3.py --no-hint --lenient sends, through a fake client (tests/fake_client.py): the
same question text, evidence bytes and other client arguments for every question, in the same order, and the same
evidence log. Nothing calls Reactor.

    python -m unittest discover tests                                       # the first 40 dev questions
    python tests/test_run_v3_test.py sample_hard.jsonl dev-large/histories.jsonl   # any set, in full
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_v3_test  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_client.py"
BASE = ["--base-url", "https://fake.invalid", "--agent-id", "agt_fake", "--timeout", "600", "--client", str(FAKE)]


def sent(log: Path) -> list[dict]:
    """The question calls in a fake client log, without what differs by design (the Reactor history id and the
    temporary evidence path)."""
    calls = []
    for line in log.read_text(encoding="utf-8").splitlines():
        call = json.loads(line)
        argv = call["argv"]
        if "--new-history" in argv:
            continue
        for name in ("--history", "--evidence"):
            if name in argv:
                argv[argv.index(name) + 1] = "<varies>"
        calls.append(call)
    return calls


def run(script: str, questions: Path, histories: Path, out: Path, *flags: str, timeout_once: str = "") -> Path:
    log = out.with_suffix(".calls.jsonl")
    env = {**os.environ, "REACTOR_ACCESS_KEY": "fake", "FAKE_CLIENT_LOG": str(log), "FAKE_TIMEOUT_ONCE": timeout_once}
    subprocess.run([sys.executable, str(ROOT / script), "--questions", str(questions), "--histories", str(histories),
                    *BASE, "--out", str(out), *flags], env=env, check=True, capture_output=True, cwd=ROOT)
    return log


def compare(questions: Path, histories: Path) -> tuple[int, list[str]]:
    """(questions compared, differences) between run_v3.py --no-hint --lenient and run_v3_test.py."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = run("run_v3.py", questions, histories, tmp / "v3.jsonl", "--no-hint", "--lenient")
        b = run("run_v3_test.py", questions, histories, tmp / "test.jsonl")
        calls_a, calls_b = sent(a), sent(b)
        problems = []
        if len(calls_a) != len(calls_b):
            problems.append(f"{len(calls_a)} questions sent by run_v3.py, {len(calls_b)} by run_v3_test.py")
        for x, y in zip(calls_a, calls_b):
            if x != y:
                problems.append(f"differs: {x['argv'][x['argv'].index('--question-id') + 1]}")
        if (tmp / "v3.evidence.jsonl").read_bytes() != (tmp / "test.evidence.jsonl").read_bytes():
            problems.append("evidence logs differ")
        return len(calls_a), problems


def first_questions(n: int, path: Path) -> Path:
    out = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    out.write_text("".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:n]), encoding="utf-8")
    return out


DEV = ROOT / "dev"


@unittest.skipUnless((DEV / "histories.jsonl").exists(), "needs dev/ (not in the repository)")
class SameAsRunV3(unittest.TestCase):
    def test_same_question_text_and_evidence(self):
        questions = first_questions(40, DEV / "questions.jsonl")
        try:
            n, problems = compare(questions, DEV / "histories.jsonl")
        finally:
            questions.unlink()
        self.assertEqual(n, 40)
        self.assertEqual(problems, [])

    def test_timeout_listens_then_asks_again_in_the_same_history(self):
        questions = first_questions(3, DEV / "questions.jsonl")
        first = json.loads(questions.read_text(encoding="utf-8").splitlines()[0])["question_id"]
        heard = []
        with tempfile.TemporaryDirectory() as tmp:
            out, log = Path(tmp) / "r.jsonl", Path(tmp) / "calls.jsonl"
            env = {"REACTOR_ACCESS_KEY": "fake", "FAKE_CLIENT_LOG": str(log), "FAKE_TIMEOUT_ONCE": first}
            with mock.patch.dict(os.environ, env), mock.patch.object(
                    run_v3_test, "wait_for_late", lambda *a: heard.append(a) or []):
                code = run_v3_test.main(["--questions", str(questions), "--histories", str(DEV / "histories.jsonl"),
                                         *BASE, "--out", str(out)])
            questions.unlink()
            self.assertEqual(code, 0)
            self.assertEqual(len(heard), 1)  # listened once, on the history the question timed out in
            asks = [json.loads(line)["argv"] for line in log.read_text(encoding="utf-8").splitlines()]
            first_asks = [a for a in asks if "--question-id" in a and a[a.index("--question-id") + 1] == first]
            self.assertEqual(len(first_asks), 2)
            self.assertEqual(first_asks[0], first_asks[1])  # asked again as it was, in the same Reactor history
            self.assertEqual(heard[0][2], first_asks[0][first_asks[0].index("--history") + 1])
            statuses = [json.loads(line)["status"] for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(statuses, ["timeout", "answered", "answered", "answered"])


if __name__ == "__main__":
    n, problems = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{n} questions compared: " + ("byte-identical question text, evidence and client arguments"
                                         if not problems else f"{len(problems)} difference(s)"))
    print("\n".join(problems[:20]))
    sys.exit(1 if problems else 0)
