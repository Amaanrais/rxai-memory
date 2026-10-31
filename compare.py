"""Two runs on the same questions, side by side: make_predictions.py on each results file, then score.py's scoring
against gold, as a table per kind of question and one line per question with the gold answer and both answers.

    python compare.py results-hard-v1.jsonl results-hard-v3.jsonl \\
        --questions sample_hard.jsonl --gold dev-large/gold.jsonl

Predictions are written next to each results file (results-X.jsonl -> predictions-X.jsonl), from its evidence log as
make_predictions.py reads it. Marks: ✓ correct, ? unjudged (score.py's matcher can't decide; the official judge
would), ✗ wrong, — abstained. Nothing calls Reactor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import make_predictions
from eval_retrieval import read_jsonl
from retrieval import is_name_question
from score import score_question, summarize

MARK = {"correct": "✓", "unjudged": "?", "wrong": "✗"}
COLUMNS = (("correct", "correct"), ("unjudged", "unjudged"), ("abstention_recall", "abst ok"),
           ("false_abstention", "false abst"), ("evidence_f1", "evid F1"))


def predictions_path(results: Path) -> Path:
    name = results.name
    return results.with_name(("predictions" + name[len("results"):]) if name.startswith("results") else
                             "predictions-" + name)


def predict(results: Path, questions: Path) -> dict[str, dict]:
    out = predictions_path(results)
    if make_predictions.main([str(results), "--questions", str(questions), "--out", str(out)]):
        sys.exit(f"make_predictions.py failed on {results}")
    return {p["question_id"]: p for p in read_jsonl(out)}


def kind(question: dict, gold: dict) -> str:
    if gold["abstain"]:
        return "gold abstain"
    if is_name_question(question["question"]):
        return "name"
    return gold["question_type"]


def cell(value) -> str:
    return "-" if value is None else f"{value:.2f}"


def shown(prediction: dict, result: dict) -> str:
    text = "—  (abstained)" if prediction["abstain"] else f"{MARK[result['answer']]}  {prediction['answer']}"
    return " ".join(text.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="first results.jsonl (e.g. results-hard-v1.jsonl)")
    parser.add_argument("b", type=Path, help="second results.jsonl (e.g. results-hard-v3.jsonl)")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True, help="the split's gold.jsonl")
    parser.add_argument("--width", type=int, default=38, help="characters per answer column (default 38)")
    args = parser.parse_args(argv)
    for path in (args.a, args.b):
        if not path.exists():
            sys.exit(f"{path} not found (has that run finished?)")

    questions = read_jsonl(args.questions)
    gold = {g["question_id"]: g for g in read_jsonl(args.gold)}
    names = [args.a.stem.removeprefix("results-"), args.b.stem.removeprefix("results-")]
    preds = [predict(args.a, args.questions), predict(args.b, args.questions)]
    scored = [[score_question(p.get(q["question_id"]), gold[q["question_id"]]) for q in questions] for p in preds]
    for rows in scored:  # group by kind of question rather than question_type
        for q, r in zip(questions, rows):
            r["question_type"] = kind(q, gold[q["question_id"]])
    tables = [summarize(rows) for rows in scored]

    width = max(len(k) for k in tables[0])
    print(f"\nA = {names[0]}, B = {names[1]}; each cell is A→B")
    print(f"{'':{width}}  {'n':>3}  " + "  ".join(f"{label:^11}" for _, label in COLUMNS))
    for key in tables[0]:
        a, b = tables[0][key], tables[1][key]
        print(f"{key:{width}}  {a['n']:>3}  " + "  ".join(f"{cell(a[c]):>5}→{cell(b[c]):<5}" for c, _ in COLUMNS))

    w = args.width
    print(f"\n{'question':<14}  {'kind':<12}  {'gold':<{w}}  {names[0]:<{w}}  {names[1]:<{w}}")
    changed = 0
    for i, q in enumerate(questions):
        g = gold[q["question_id"]]
        cells = [shown(p.get(q["question_id"], {"abstain": True}), rows[i]) for p, rows in zip(preds, scored)]
        flag = "" if scored[0][i]["answer"] == scored[1][i]["answer"] else "  *"
        changed += bool(flag)
        target = "(abstain)" if g["abstain"] else " ".join(g["answer"].split())
        print(f"{q['question_id']:<14}  {kind(q, g):<12.12}  {target[:w]:<{w}}  {cells[0][:w]:<{w}}  "
              f"{cells[1][:w]:<{w}}{flag}")
        print(f"{'':<14}  {q['question'][:110]}")
    print(f"\n* {changed} question(s) scored differently. ✓ correct  ? unjudged  ✗ wrong  — abstained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
