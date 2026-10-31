"""What two test runs sent, side by side: each question's whole message (reactor/client.py's build_message on the
question text and evidence in the run's evidence log) against its full history, per question and per dataset history
against loading the history once. Nothing calls Reactor.

    python efficiency.py results-test.evidence.jsonl results-test-v3.evidence.jsonl \\
        --questions test/questions.jsonl --histories test/histories.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from eval_retrieval import read_jsonl, stream_histories
from run import ROOT, prompt
from run_v2 import prompt_v2

sys.path.insert(0, str(ROOT / "reactor"))
import client  # noqa: E402  (the kit's client, imported only for build_message)


def history_chars(history: dict) -> int:
    """A history's size in characters, as app.py counts it: each session's id and date, each turn's role and text."""
    return sum(len(s["session_id"]) + len(s["date"]) + 4
               + sum(len(t["role"]) + 2 + len(t["content"]) + 1 for t in s["turns"]) for s in history["sessions"])


def message_chars(log: Path, questions: dict[str, dict]) -> dict[str, int]:
    """Each question's whole message in characters, from the first evidence-log line for it (a retry sends the same)."""
    sizes = {}
    for line in read_jsonl(log):
        if line["question_id"] in sizes:
            continue
        q = questions[line["question_id"]]
        if line.get("version", 1) == 1:
            text = prompt(q["question"])
        else:
            missing = line["missing"] if line.get("hint", True) else []
            text = prompt_v2(q["question"], missing, line.get("lenient", False))
        passages = client.load_evidence(json.dumps(line["evidence"])) if line["evidence"] else []
        message = client.build_message(text, passages, "00000000", client.parse_question_date(q["question_date"]))
        sizes[line["question_id"]] = len(message)
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", type=Path, nargs="+", help="evidence logs (results-X.evidence.jsonl)")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    args = parser.parse_args(argv)

    questions = {q["question_id"]: q for q in read_jsonl(args.questions)}
    full = {h["history_id"]: history_chars(h)
            for h in stream_histories(args.histories, {q["history_id"] for q in questions.values()})}
    runs = {log.name.removeprefix("results-").removesuffix(".evidence.jsonl"): message_chars(log, questions)
            for log in args.logs}
    totals = {}
    for name, sizes in runs.items():
        totals[name] = {}
        for qid, n in sizes.items():
            h = questions[qid]["history_id"]
            totals[name][h] = totals[name].get(h, 0) + n

    print(f"{len(questions)} questions in {len(full)} histories; a full history is "
          f"{statistics.median(full.values()) / 1e6:.2f}M characters (median; {min(full.values()) / 1e6:.2f}M to "
          f"{max(full.values()) / 1e6:.2f}M)\n")
    print(f"{'':38}" + "".join(f"{name:>12}" for name in runs))
    def less(sizes: dict[str, int], history_of) -> list[float]:
        return [full[history_of(key)] / n for key, n in sizes.items()]

    rows = [
        ("questions in the log", lambda s, t: f"{len(s)}"),
        ("message, median characters", lambda s, t: f"{statistics.median(s.values()) / 1e3:.1f}k"),
        ("message, max characters", lambda s, t: f"{max(s.values()) / 1e3:.1f}k"),
        ("per question, x less than history",
         lambda s, t: f"{statistics.median(less(s, lambda q: questions[q]['history_id'])):.0f}x"),
        ("per history, characters sent", lambda s, t: f"{statistics.median(t.values()) / 1e3:.0f}k"),
        ("per history, x less than loading", lambda s, t: f"{statistics.median(less(t, str)):.0f}x"),
        ("per history, range", lambda s, t: f"{min(less(t, str)):.0f}-{max(less(t, str)):.0f}x"),
    ]
    for label, cell in rows:
        print(f"{label:38}" + "".join(f"{cell(runs[name], totals[name]):>12}" for name in runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
