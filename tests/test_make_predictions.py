"""make_predictions.py: one prediction per question, the original's late answer first, the earlier of equals.

    python -m unittest discover tests
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from make_predictions import choose, main, recover_late  # noqa: E402
from score import validate  # noqa: E402

EVIDENCE = [{"id": "s_aaa", "date": "2023/05/01 (Mon) 10:00", "text": "user: the answer is Hawaii"},
            {"id": "s_bbb", "date": "2023/05/02 (Tue) 10:00", "text": "user: something else"}]


def timeout(qid: str, ref: str) -> dict:
    return {"status": "timeout", "question_id": qid, "history_id": "R1", "ref": ref, "late_replies": []}


def answered(qid: str, ref: str, answer: str, late: list[dict] | None = None) -> dict:
    return {"status": "answered", "question_id": qid, "history_id": "R1", "ref": ref, "answer": answer,
            "reply": "", "citations": [], "abstained": False, "late_replies": late or []}


def late(ref: str, answer: str) -> dict:
    return {"ref": ref, "answer": answer, "cycle": 7}


class ChooseTest(unittest.TestCase):
    def test_original_late_answer_beats_answered_retry(self):
        lines = [timeout("q_a", "r1"), answered("q_a", "r2", "retry [E2]"),
                 answered("q_b", "r3", "b [E1]", late=[late("r1", "original [E1]")])]
        chosen = choose(lines)
        self.assertEqual(chosen["q_a"]["answer"], "original [E1]")
        self.assertTrue(chosen["q_a"]["recovered"])

    def test_original_beats_retry_when_both_answer_late(self):
        lines = [timeout("q_a", "r1"), timeout("q_a", "r2"),
                 answered("q_b", "r3", "b", late=[late("r2", "retry's late"), late("r1", "original's late")])]
        self.assertEqual(choose(lines)["q_a"]["answer"], "original's late")

    def test_equally_valid_lines_keep_the_earlier(self):
        lines = [answered("q_a", "r1", "first"), answered("q_a", "r2", "second")]
        self.assertEqual(choose(lines)["q_a"]["answer"], "first")

    def test_recorded_recovery_is_not_recovered_twice(self):
        lines = [timeout("q_a", "r1"), answered("q_b", "r2", "b", late=[late("r1", "original")])]
        lines += recover_late(lines)  # what run.py records
        self.assertEqual(len([r for r in lines if r.get("recovered")]), 1)
        self.assertEqual(recover_late(lines, answered_too=True), [])
        self.assertEqual(choose(lines)["q_a"]["answer"], "original")

    def test_no_answer_keeps_the_latest_line(self):
        lines = [timeout("q_a", "r1"), {"status": "error", "question_id": "q_a", "error": "x"}]
        self.assertEqual(choose(lines)["q_a"]["status"], "error")


class MainTest(unittest.TestCase):
    def test_predictions_have_unique_ids_and_the_original_answer(self):
        lines = [timeout("q_a", "r1"), answered("q_a", "r2", "retry [E2]"),
                 answered("q_b", "r3", "b [E2]", late=[late("r1", "Hawaii [E1]")]),
                 answered("q_b", "r4", "b again [E1]"), timeout("q_c", "r5")]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
            (tmp / "results.evidence.jsonl").write_text(
                "".join(json.dumps({"question_id": q, "evidence": EVIDENCE}) + "\n" for q in ("q_a", "q_b", "q_c")),
                encoding="utf-8")
            (tmp / "questions.jsonl").write_text(
                "".join(json.dumps({"question_id": q, "history_id": "h_1"}) + "\n" for q in ("q_a", "q_b", "q_c")),
                encoding="utf-8")
            self.assertEqual(main([str(tmp / "results.jsonl"), "--questions", str(tmp / "questions.jsonl"),
                                   "--out", str(tmp / "predictions.jsonl")]), 0)
            predictions = [json.loads(line) for line in (tmp / "predictions.jsonl").read_text().splitlines()]
        ids = [p["question_id"] for p in predictions]
        self.assertEqual(sorted(ids), ["q_a", "q_b", "q_c"])
        self.assertEqual(validate(predictions, set(ids)), [])
        by_id = {p["question_id"]: p for p in predictions}
        self.assertEqual(by_id["q_a"], {"question_id": "q_a", "abstain": False, "answer": "Hawaii",
                                        "citations": ["s_aaa"]})
        self.assertEqual(by_id["q_b"]["answer"], "b")  # the earlier of two answered lines
        self.assertEqual(by_id["q_c"], {"question_id": "q_c", "abstain": True, "answer": "", "citations": []})


if __name__ == "__main__":
    unittest.main()
