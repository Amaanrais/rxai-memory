"""results.jsonl (from run.py) -> predictions.jsonl in the submission format, checked with score.py's validator.

    python make_predictions.py results.jsonl --questions dev/questions.jsonl --out predictions.jsonl

Citations, first that gives any session id:
  1. the [E#] labels on the agent's FINAL ANSWER line;
  2. the [E#] labels in the last paragraph of its reasoning (the reply before the FINAL ANSWER line);
  3. suggest_citations(), as run.py logged it.
Labels map to session ids through the excerpts run.py sent (<results>.evidence.jsonl): E1 is the first excerpt.
A question abstains when the agent said INSUFFICIENT_EVIDENCE or its status isn't "answered"; it keeps the
suggested citations, since evidence F1 counts on every answerable question, abstained or not. Questions with no
result line abstain too. When a question has several result lines, an answered one wins, then the latest.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from eval_retrieval import read_jsonl
from score import validate, write_jsonl

# the client's final-answer marker (reactor/client.py, _FINAL)
FINAL = re.compile(r"^[ \t]*#{1,3}[ \t]*final[ _-]?answer[ \t]*\[[ \t]*ref[ \t]*:[ \t]*[0-9a-f]{8}[ \t]*\]",
                   re.IGNORECASE | re.MULTILINE)
GROUP = re.compile(r"\[([^\[\]]*)\]")
LABEL = re.compile(r"\bE(\d+)\b")
LABELS = re.compile(r"\s*\[[^\[\]]*\bE\d+\b[^\[\]]*\]")


def labels(text: str) -> list[int]:
    """The [E#] numbers in ``text``, in order, without repeats ([E1, E3] too)."""
    found = [int(n) for group in GROUP.findall(text) for n in LABEL.findall(group)]
    return list(dict.fromkeys(found))


def to_ids(numbers: list[int], evidence: list[dict]) -> list[str]:
    ids = [evidence[n - 1]["id"] for n in numbers if 1 <= n <= len(evidence)]
    return list(dict.fromkeys(i for i in ids if i))


def last_paragraph(reply: str) -> str:
    """The last paragraph of the reasoning: the reply up to its FINAL ANSWER line."""
    mark = FINAL.search(reply)
    reasoning = reply[:mark.start()] if mark else reply
    paragraphs = [p for p in re.split(r"\n\s*\n", reasoning) if p.strip()]
    return paragraphs[-1] if paragraphs else ""


def predict(question_id: str, result: dict | None, logged: dict | None) -> tuple[dict, str]:
    """(prediction, where its citations came from)."""
    evidence = (logged or {}).get("evidence", [])
    suggested = (logged or {}).get("suggested", [])
    if not result or result.get("status") != "answered" or result.get("abstained"):
        return {"question_id": question_id, "abstain": True, "answer": "", "citations": suggested}, "abstain"
    answer = result.get("answer", "")
    for source, text in (("answer", answer), ("reasoning", last_paragraph(result.get("reply", "")))):
        cited = to_ids(labels(text), evidence)
        if not cited and source == "answer" and not evidence:  # no log entry: the client's own mapping
            cited = list(dict.fromkeys(c["id"] for c in result.get("citations", []) if c.get("id")))
        if cited:
            break
    else:
        source, cited = "suggested", suggested
    clean = LABELS.sub("", answer).strip()
    return {"question_id": question_id, "abstain": False, "answer": clean, "citations": cited}, source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--questions", type=Path, required=True, help="the split's questions.jsonl")
    parser.add_argument("--evidence-log", type=Path, help="default: <results>.evidence.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    log_path = args.evidence_log or args.results.with_suffix(".evidence.jsonl")

    question_ids = [q["question_id"] for q in read_jsonl(args.questions)]
    results: dict[str, dict] = {}
    for r in read_jsonl(args.results):
        qid = r.get("question_id")
        if qid and (r.get("status") == "answered" or results.get(qid, {}).get("status") != "answered"):
            results[qid] = r
    logged = {e["question_id"]: e for e in read_jsonl(log_path)} if log_path.exists() else {}
    if not logged:
        print(f"warning: no evidence log at {log_path}: labels map through the client's citations only",
              file=sys.stderr)

    predictions, sources = [], Counter()
    for qid in question_ids:
        prediction, source = predict(qid, results.get(qid), logged.get(qid))
        predictions.append(prediction)
        sources["no result" if qid not in results else source] += 1
    errors = validate(predictions, set(question_ids))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    write_jsonl(args.out, predictions)
    print(f"{len(predictions)} prediction(s) written to {args.out}; citations from: "
          + ", ".join(f"{k} {v}" for k, v in sources.most_common()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
