"""Rebuild the dev or dev-large split from the public LongMemEval files, identical to the released ones.

dev comes from longmemeval_s_cleaned.json, dev-large from longmemeval_m_cleaned.json
(https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). Standard library only.
Usage: python convert_longmemeval.py longmemeval_m_cleaned.json dev-large
"""
import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path

CHUNK = 16 * 1024 * 1024


def iter_array(path):
    """Yield the elements of a top-level JSON array without loading the whole file."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(CHUNK)
        pos = buf.index("[") + 1
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(CHUNK)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(CHUNK)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            buf, pos = buf[end:], 0


def read_jsonl(path):
    """Yield dicts from a JSON lines file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


DATE_FMT = "%Y/%m/%d (%a) %H:%M"
DROP_TYPES = {"single-session-preference"}


def opaque_id(prefix, raw, salt):
    """Deterministic opaque id: prefix plus a salted hash of the raw id."""
    return prefix + hashlib.sha256(f"{salt}:{prefix}{raw}".encode()).hexdigest()[:12]


def convert_sessions(rec, salt):
    """Sessions of one record in date order, repeated session ids dropped."""
    triples = zip(rec["haystack_session_ids"], rec["haystack_dates"], rec["haystack_sessions"])
    seen, sessions = set(), []
    for sid, date, turns in sorted(triples, key=lambda t: datetime.strptime(t[1], DATE_FMT)):
        if sid in seen:
            continue
        seen.add(sid)
        sessions.append({
            "session_id": opaque_id("s_", sid, salt),
            "date": date,
            "turns": [{"turn_id": opaque_id("t_", f"{sid}:{i}", salt), "role": t["role"], "content": t["content"]}
                      for i, t in enumerate(turns)],
        })
    return sessions


def question_date(rec):
    """Question date moved to just after the last session if a session is dated later; the day never changes."""
    asked = datetime.strptime(rec["question_date"], DATE_FMT)
    last = max(datetime.strptime(d, DATE_FMT) for d in rec["haystack_dates"])
    if last <= asked:
        return rec["question_date"]
    if last.date() != asked.date():
        raise ValueError(f"{rec['question_id']}: session dated on a later day than the question")
    moved = last + timedelta(minutes=1)
    return (moved if moved.date() == last.date() else last).strftime(DATE_FMT)


def convert_record(rec, salt):
    """Map one LongMemEval record to (history, question, gold)."""
    qid = rec["question_id"]
    abstain = qid.endswith("_abs")
    history_id = opaque_id("h_", qid, salt)
    question_id = opaque_id("q_", qid, salt)
    history = {"history_id": history_id, "sessions": convert_sessions(rec, salt)}
    question = {"question_id": question_id, "history_id": history_id,
                "question": rec["question"], "question_date": question_date(rec)}
    gold = {"question_id": question_id, "question_type": rec["question_type"], "abstain": abstain,
            "answer": str(rec["answer"]), "aliases": [],
            "evidence": [] if abstain else [opaque_id("s_", s, salt) for s in rec["answer_session_ids"]]}
    return history, question, gold


def variant(question, gold, raw_qid, text, salt):
    """Question and gold for a rephrasing: same history, date and gold as the original, its own question id."""
    qid = opaque_id("q_", f"{raw_qid}#uncued", salt)
    return {**question, "question_id": qid, "question": text}, {**gold, "question_id": qid}


def keep(rec):
    """Whether a record's question type is part of the release."""
    return rec["question_type"] not in DROP_TYPES


def convert(records, salt):
    """Yield (history, question, gold) for the kept records."""
    for rec in records:
        if keep(rec):
            yield convert_record(rec, salt)


NAMES = ("histories", "questions", "gold")
REPHRASINGS = Path(__file__).resolve().parent / "uncued_assistant.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--rephrasings", type=Path, default=REPHRASINGS)
    args = ap.parse_args()
    rephrased = {r["question_id"]: r["question"] for r in read_jsonl(args.rephrasings)}
    salt = os.environ.get("LMEHACK_SALT", "dev")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Files are written as .part and renamed only after the last record, so a crash leaves no valid-looking split.
    parts = {k: out / f"{k}.jsonl.part" for k in NAMES}
    n = repeats = moved = variants = 0
    with ExitStack() as stack:
        files = {k: stack.enter_context(open(p, "w", encoding="utf-8")) for k, p in parts.items()}
        for rec in iter_array(args.src):
            if not keep(rec):
                continue
            history, question, gold = convert_record(rec, salt)
            for k, r in (("histories", history), ("questions", question), ("gold", gold)):
                files[k].write(json.dumps(r, ensure_ascii=False) + "\n")
            if rec["question_id"] in rephrased:
                for k, r in zip(("questions", "gold"), variant(question, gold, rec["question_id"],
                                                               rephrased[rec["question_id"]], salt)):
                    files[k].write(json.dumps(r, ensure_ascii=False) + "\n")
                variants += 1
            repeats += len(rec["haystack_session_ids"]) - len(history["sessions"])
            moved += question["question_date"] != rec["question_date"]
            n += 1
    for k, p in parts.items():
        p.replace(out / f"{k}.jsonl")
    print(f"{n} questions and {variants} uncued rephrasings written to {out}; {repeats} repeated sessions dropped; {moved} question dates moved after the last session")


if __name__ == "__main__":
    main()
