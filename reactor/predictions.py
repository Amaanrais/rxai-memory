"""Turns client.py results into the evaluator's predictions.jsonl -- one line per question:

    {"question_id": "q_3f9a...", "abstain": false, "answer": "Hawaii", "citations": ["s_81c2...", "s_07de..."]}

Collect one client line per question, then convert (Python 3.11+, nothing to install):

    python client.py ... --question-id q_3f9a... --evidence passages.json >> results.jsonl
    python predictions.py results.jsonl --histories histories.jsonl --out predictions.jsonl

Mapping rules -- nothing is invented, and whatever doesn't map is reported on stderr:
- ``question_id``: the one you gave the client (``--question-id``). A result without one is skipped.
- ``citations``: the evaluator wants **session ids from histories.jsonl**, so give each passage you send that
  session's id as its ``"id"``. Citations are the ids of the passages the answer cites, in the order the client
  lists them, without repeats. A cited passage you gave no id is left out, and so (with ``--histories``) is an id
  that isn't a session id there. Labels the agent made up (``unknown_citations``) are never mapped.
- ``answer``: the final-answer text, without its ``[E#]`` citation labels.
- Abstaining is **your** choice, never the adapter's. By default, a question the agent answered
  ``INSUFFICIENT_EVIDENCE`` and a question without an answer (timeout, closed, refused, error) are skipped.
  ``--abstain-on agent`` writes the first kind as ``{"abstain": true, "answer": "", "citations": []}``,
  ``--abstain-on no-answer`` the second, ``--abstain-on agent,no-answer`` both.
- One prediction per question: two results for one ``question_id`` stop the conversion. Keep the one you mean.

``--histories`` reads the session ids from the ``session_id`` fields of histories.jsonl, at any depth
(``--session-key`` if the field is named otherwise). Exit 0: every result became a prediction. 1: some were
skipped (listed); the file holds the rest. 2: bad input or duplicates; nothing written.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import sys
from pathlib import Path

ABSTAIN_ON = ("agent", "no-answer")
NO_ANSWER = ("timeout", "closed", "refused", "error")
_LABELS = re.compile(r"\s*\[\s*E\d+(?:\s*[,;]\s*E\d+)*\s*\]")


def strip_labels(answer: str) -> str:
    """The answer without its ``[E#]``/``[E1, E3]`` citation labels, spacing tidied."""

    text = _LABELS.sub("", answer)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def session_ids(lines: list[str], key: str = "session_id") -> set[str]:
    """Every string value of ``key``, at any depth, in histories.jsonl's lines."""

    found: set[str] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key and isinstance(v, str):
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for line in lines:
        if line.strip():
            walk(json.loads(line))
    return found


def convert(results: list[dict], abstain_on: set[str], known: set[str] | None = None
            ) -> tuple[list[dict], list[str], list[str]]:
    """``(predictions, skipped, notes)``: a prediction per usable result, in order; ``skipped`` says why each other
    result wasn't one; ``notes`` reports citations left out. Raises ValueError on a repeated question_id."""

    seen: dict[str, int] = {}
    for n, r in enumerate(results, 1):
        qid = r.get("question_id")
        if isinstance(qid, str) and qid:
            if qid in seen:
                raise ValueError(f"question {qid!r}: results on lines {seen[qid]} and {n}; keep one")
            seen[qid] = n
    predictions, skipped, notes = [], [], []
    for n, r in enumerate(results, 1):
        qid, status = r.get("question_id"), r.get("status")
        where = f"line {n}" + (f" ({qid})" if qid else "")
        if not isinstance(qid, str) or not qid:
            skipped.append(f"{where}: no question_id (run the client with --question-id)")
            continue
        if status == "answered" and r.get("abstained"):
            if "agent" in abstain_on:
                predictions.append({"question_id": qid, "abstain": True, "answer": "", "citations": []})
            else:
                skipped.append(f"{where}: the agent abstained (INSUFFICIENT_EVIDENCE); --abstain-on agent to abstain")
            continue
        if status in NO_ANSWER:
            if "no-answer" in abstain_on:
                predictions.append({"question_id": qid, "abstain": True, "answer": "", "citations": []})
            else:
                skipped.append(f"{where}: no answer ({status}{': ' + str(r['code']) if r.get('code') else ''}); "
                               "--abstain-on no-answer to abstain")
            continue
        if status != "answered" or not isinstance(r.get("answer"), str):
            skipped.append(f"{where}: not a --question result (status {status!r})")
            continue
        answer = strip_labels(r["answer"])
        if not answer:
            skipped.append(f"{where}: the answer is empty without its citation labels")
            continue
        cited: list[str] = []
        for c in r.get("citations") or []:
            cid = c.get("id") if isinstance(c, dict) else None
            label = c.get("label") if isinstance(c, dict) else c
            if not isinstance(cid, str) or not cid:
                notes.append(f"{where}: {label} cites a passage you gave no id: left out")
            elif known is not None and cid not in known:
                notes.append(f"{where}: {label} -> {cid!r} isn't a session id in histories.jsonl: left out")
            elif cid not in cited:
                cited.append(cid)
        for label in r.get("unknown_citations") or []:
            notes.append(f"{where}: {label} wasn't among the passages sent: left out")
        predictions.append({"question_id": qid, "abstain": False, "answer": answer, "citations": cited})
    return predictions, skipped, notes


def _abstain_on(value: str) -> set[str]:
    chosen = {v.strip() for v in value.split(",") if v.strip()}
    if not chosen <= set(ABSTAIN_ON):
        raise argparse.ArgumentTypeError(f"one or both of: {', '.join(ABSTAIN_ON)}")
    return chosen


def read_text(path: Path) -> str:
    """A text file as UTF-8 (with or without a BOM), or as UTF-16 when it starts with a UTF-16 BOM -- what
    Windows PowerShell 5.1's ``>>`` writes. Anything else is a ValueError that says so."""

    raw = path.read_bytes()
    try:
        if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return raw.decode("utf-16")
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not UTF-8 or UTF-16 text ({exc.reason} at byte {exc.start})") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="client.py results -> the evaluator's predictions.jsonl")
    parser.add_argument("results", type=Path, help="client.py output, one JSON line per question")
    parser.add_argument("--out", type=Path, required=True, help="predictions.jsonl to write")
    parser.add_argument("--histories", type=Path, help="histories.jsonl: cited ids must be its session ids")
    parser.add_argument("--session-key", default="session_id", help="the session id's field in histories.jsonl")
    parser.add_argument("--abstain-on", type=_abstain_on, default=set(),
                        help="agent and/or no-answer (comma-separated): write those as abstentions; default: skip")
    args = parser.parse_args(argv)
    try:
        results = []
        for n, line in enumerate(read_text(args.results).splitlines(), 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{args.results} line {n}: not a JSON object")
                results.append(obj)
        known = None
        if args.histories:
            known = session_ids(read_text(args.histories).splitlines(), args.session_key)
            if not known:
                raise ValueError(f"{args.histories}: no {args.session_key!r} field found (see --session-key)")
        predictions, skipped, notes = convert(results, args.abstain_on, known)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8")
    os.replace(tmp, args.out)
    for line in skipped:
        print(f"skipped: {line}", file=sys.stderr)
    for line in notes:
        print(f"note: {line}", file=sys.stderr)
    abstained = sum(p["abstain"] for p in predictions)
    print(f"{len(predictions)} prediction(s) ({abstained} abstained) written to {args.out}; "
          f"{len(skipped)} result(s) skipped", file=sys.stderr)
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
