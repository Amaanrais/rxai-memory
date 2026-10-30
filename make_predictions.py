"""results.jsonl (from run.py) -> predictions.jsonl in the submission format, checked with score.py's validator.

    python make_predictions.py results.jsonl --questions dev/questions.jsonl --out predictions.jsonl

Citations, first that gives any session id:
  1. the [E#] labels on the agent's FINAL ANSWER line;
  2. the [E#] labels in the last paragraph of its reasoning (the reply before the FINAL ANSWER line);
  3. support: the excerpts we sent whose text contains the agent's answer (normalized as score.py does), else
     those holding at least SUPPORT_SHARE of the answer's content words; else nothing. A match in more than
     MAX_MATCHED of the excerpts singles none out (a bare "2"), so it cites nothing either. Never the
     top-ranked session by default: the organisers' rule is that fallback citations must support the answer.
Labels and excerpts map to session ids through what run.py sent (<results>.evidence.jsonl): E1 is the first
excerpt. A question abstains, with no citations, when the agent said INSUFFICIENT_EVIDENCE or its status isn't
"answered", or when it has no result line. When a question has several result lines, an answered one wins, then
the latest. A timed-out question whose answer arrived later, in another line's late_replies (Reactor's own answer
to its ref), counts as answered with that answer (``recover_late()``; run.py also records such lines).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from eval_retrieval import read_jsonl
from retrieval import STOPWORDS, TIME_CUE_LABEL
from score import normalize, validate, write_jsonl

# the client's final-answer marker (reactor/client.py, _FINAL)
FINAL = re.compile(r"^[ \t]*#{1,3}[ \t]*final[ _-]?answer[ \t]*\[[ \t]*ref[ \t]*:[ \t]*[0-9a-f]{8}[ \t]*\]",
                   re.IGNORECASE | re.MULTILINE)
GROUP = re.compile(r"\[([^\[\]]*)\]")
LABEL = re.compile(r"\bE(\d+)\b")
LABELS = re.compile(r"\s*\[[^\[\]]*\bE\d+\b[^\[\]]*\]")
# lines retrieval.evidence() adds to an excerpt, not words from the history
NOTE = re.compile(r"^(?:\(\d+ days? before the question date\)|\[most recent\])$", re.MULTILINE)
SUPPORT_SHARE = 0.6
# the client's abstention: INSUFFICIENT_EVIDENCE as the whole answer or its first word (reactor/client.py)
INSUFFICIENT = re.compile(r"^\W*INSUFFICIENT_EVIDENCE\b", re.IGNORECASE)
# a match in more than this share of the excerpts doesn't single any out ("2" is in all of them): no support
MAX_MATCHED = 0.5


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


def recovered(timed_out: dict, late: dict, received_in: dict) -> dict:
    """An answered result line for a timed-out question, from Reactor's own final answer to its ref that arrived
    late, in ``received_in``'s ``late_replies``."""
    return {"status": "answered", "question_id": timed_out["question_id"], "history_id": timed_out.get("history_id"),
            "ref": late["ref"], "question": timed_out.get("question"), "question_date": timed_out.get("question_date"),
            "answer": late.get("answer", ""), "reply": "", "citations": [],
            "abstained": bool(INSUFFICIENT.match(late.get("answer", ""))),
            "recovered": {"from_question_id": received_in.get("question_id"), "cycle": late.get("cycle")}}


def recover_late(lines: list[dict]) -> list[dict]:
    """Recovered lines (``recovered()``) for timed-out questions with no answered line whose answer arrived in a
    later line's ``late_replies``, matched by the timed-out request's ref."""
    answered = {r.get("question_id") for r in lines if r.get("status") == "answered"}
    timed_out: dict[str, dict] = {}  # ref -> the timed-out line, from earlier lines only
    out = []
    for r in lines:
        for late in r.get("late_replies") or []:
            line = timed_out.get(late.get("ref"))
            if line and line["question_id"] not in answered:
                out.append(recovered(line, late, r))
                answered.add(line["question_id"])
        if r.get("status") == "timeout" and r.get("ref") and r.get("question_id"):
            timed_out[r["ref"]] = r
    return out


def excerpt_words(text: str) -> str:
    """An excerpt's history text, normalized as score.py does, padded for whole-phrase matching."""
    text = NOTE.sub("", text).replace(TIME_CUE_LABEL, "")
    return f" {normalize(text)} "


def supported(answer: str, evidence: list[dict]) -> tuple[list[str], str]:
    """(session ids of the excerpts that support ``answer``, how): those containing the whole normalized answer
    ("exact"), else those holding at least SUPPORT_SHARE of its content words ("partial"), else none. Matches
    in more than MAX_MATCHED of the excerpts don't single any out and count as none."""
    phrase = normalize(answer)
    if not phrase:
        return [], "unsupported"
    texts = [(e["id"], excerpt_words(e["text"])) for e in evidence if e.get("id")]
    singles_out = lambda matched: 0 < len(matched) <= MAX_MATCHED * len(texts)
    exact = [sid for sid, text in texts if f" {phrase} " in text]
    if exact:
        return (list(dict.fromkeys(exact)), "exact") if singles_out(exact) else ([], "unspecific")
    words = {w for w in phrase.split() if w not in STOPWORDS}
    if words:
        partial = [sid for sid, text in texts
                   if len(words & set(text.split())) >= SUPPORT_SHARE * len(words)]
        if partial:
            return (list(dict.fromkeys(partial)), "partial") if singles_out(partial) else ([], "unspecific")
    return [], "unsupported"


def predict(question_id: str, result: dict | None, logged: dict | None) -> tuple[dict, str]:
    """(prediction, where its citations came from)."""
    evidence = (logged or {}).get("evidence", [])
    if not result or result.get("status") != "answered" or result.get("abstained"):
        return {"question_id": question_id, "abstain": True, "answer": "", "citations": []}, "abstain"
    answer = result.get("answer", "")
    clean = LABELS.sub("", answer).strip()
    for source, text in (("answer labels", answer), ("reasoning labels", last_paragraph(result.get("reply", "")))):
        cited = to_ids(labels(text), evidence)
        if not cited and source == "answer labels" and not evidence:  # no log entry: the client's own mapping
            cited = list(dict.fromkeys(c["id"] for c in result.get("citations", []) if c.get("id")))
        if cited:
            break
    else:
        cited, how = supported(clean, evidence)
        source = f"support ({how})"
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
    lines = read_jsonl(args.results)
    results: dict[str, dict] = {}
    for r in lines + recover_late(lines):
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
        if qid in results and results[qid].get("recovered"):
            source = f"late reply, {source}"
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
