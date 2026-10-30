"""Score a predictions.jsonl against a split's gold.jsonl and print a table per question type.

Self-scoring on dev: answers are matched against the gold answer and its accepted variants; answers the matcher cannot
decide are "unjudged". Official test scoring additionally sends every answer to an LLM judge.
Standard library only. Usage: python score.py dev/gold.jsonl predictions.jsonl [--details results.jsonl]
"""
import argparse
import json
import re
from pathlib import Path


def read_jsonl(path):
    """Yield dicts from a JSON lines file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    """Write an iterable of dicts as JSON lines, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



FIELDS = {"question_id": str, "abstain": bool, "answer": str, "citations": list}
NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
NUMBERS.update({w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)})
# A token is a number with its inner separators (2.5, 1:25:50, 2,000, 1/48) or a word with its hyphens.
TOKEN = re.compile(r"\d+(?:[.,:/]\d+)*|[^\W\d_]+(?:-[^\W\d_]+)*")
THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")
ARTICLES = {"a", "an", "the"}
ALSO_OK = re.compile(r"^(.+?)\.\s+(.+?)\s*\(including the last day\) is also acceptable\.?$", re.I)
OR_PAREN = re.compile(r"^(.+?)\s*\(or (.+?)\)\s*$", re.I)


def validate(predictions, gold_ids):
    """Format errors in a list of prediction dicts, as readable strings."""
    errors, seen = [], set()
    for i, p in enumerate(predictions, 1):
        if not isinstance(p, dict):
            errors.append(f"line {i}: not a JSON object")
            continue
        for field, kind in FIELDS.items():
            if not isinstance(p.get(field), kind):
                errors.append(f"line {i}: {field} missing or not {kind.__name__}")
        if isinstance(p.get("citations"), list) and not all(isinstance(c, str) for c in p["citations"]):
            errors.append(f"line {i}: citations must be strings")
        qid = p.get("question_id")
        if qid in seen:
            errors.append(f"line {i}: duplicate question_id {qid}")
        elif isinstance(qid, str) and qid not in gold_ids:
            errors.append(f"line {i}: unknown question_id {qid}")
        seen.add(qid)
    return errors


def normalize(text):
    """Lowercase, number words as digits, punctuation and articles removed, whitespace collapsed."""
    out = []
    for tok in TOKEN.findall(text.lower()):
        if tok[0].isdigit():
            out.append(tok.replace(",", "") if THOUSANDS.fullmatch(tok) else tok)
        elif all(part in NUMBERS for part in tok.split("-")):
            out.append(str(sum(NUMBERS[part] for part in tok.split("-"))))
        else:
            out.extend(w for w in tok.split("-") if w not in ARTICLES)
    return " ".join(out)


def aliases(answer):
    """Acceptable answer strings derived from a gold answer, the answer itself first."""
    for pattern in (ALSO_OK, OR_PAREN):
        m = pattern.match(answer.strip())
        if m:
            return [answer, m.group(1), m.group(2)]
    return [answer]


def answer_matched(pred_answer, gold):
    """Whether a normalized gold answer or alias occurs as a whole phrase in the prediction."""
    pred = f" {normalize(pred_answer)} "
    candidates = aliases(gold["answer"]) + gold.get("aliases", [])
    return any(f" {c} " in pred for c in map(normalize, candidates) if c)


def evidence_f1(citations, evidence):
    """F1 of cited vs gold session ids."""
    cited, gold = set(citations), set(evidence)
    hit = len(cited & gold)
    if not hit:
        return 0.0
    precision, recall = hit / len(cited), hit / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_question(pred, gold):
    """Per-question result: answer status (correct, wrong, unjudged), abstention correctness, evidence F1."""
    pred = pred or {"abstain": False, "answer": "", "citations": []}
    if gold["abstain"] or pred["abstain"]:
        status = "correct" if gold["abstain"] == pred["abstain"] else "wrong"
    elif not pred["answer"].strip():
        status = "wrong"
    else:
        status = "correct" if answer_matched(pred["answer"], gold) else "unjudged"
    return {"question_id": gold["question_id"], "question_type": gold["question_type"],
            "gold_abstain": gold["abstain"], "answer": status,
            "abstain_ok": gold["abstain"] == pred["abstain"],
            "evidence_f1": None if gold["abstain"] else evidence_f1(pred["citations"], gold["evidence"])}


def summarize(results):
    """Per-type and overall aggregates of per-question results."""
    groups = {}
    for r in results:
        groups.setdefault(r["question_type"], []).append(r)
    groups = dict(sorted(groups.items()))
    groups["all"] = list(results)
    rows = {}
    for name, rs in groups.items():
        f1s = [r["evidence_f1"] for r in rs if r["evidence_f1"] is not None]
        abst = [r for r in rs if r["gold_abstain"]]
        answerable = [r for r in rs if not r["gold_abstain"]]
        rows[name] = {
            "n": len(rs),
            "correct": sum(r["answer"] == "correct" for r in rs) / len(rs),
            "unjudged": sum(r["answer"] == "unjudged" for r in rs) / len(rs),
            "abstention_n": len(abst),
            "abstention_recall": sum(r["abstain_ok"] for r in abst) / len(abst) if abst else None,
            "false_abstention": sum(not r["abstain_ok"] for r in answerable) / len(answerable) if answerable else None,
            "evidence_f1": sum(f1s) / len(f1s) if f1s else None,
        }
    return rows


COLUMNS = ("n", "correct", "unjudged", "abstention_n", "abstention_recall", "false_abstention", "evidence_f1")


def fmt(v):
    """Table cell: floats to three decimals, None as a dash."""
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("predictions")
    ap.add_argument("--details", help="write per-question results to this JSONL file")
    args = ap.parse_args()
    gold = {g["question_id"]: g for g in read_jsonl(args.gold)}
    try:
        preds = list(read_jsonl(args.predictions))
    except json.JSONDecodeError as e:
        raise SystemExit(f"predictions are not valid JSON lines: {e}")
    errors = validate(preds, set(gold))
    if errors:
        raise SystemExit("\n".join(errors))
    by_id = {p["question_id"]: p for p in preds}
    missing = len(gold) - len(by_id)
    results = [score_question(by_id.get(qid), g) for qid, g in gold.items()]
    if args.details:
        write_jsonl(args.details, results)
    if missing:
        print(f"{missing} questions without a prediction, scored as wrong")
    width = max(map(len, ["type", *summarize(results)]))
    print(f"{'type':{width}s}  " + "  ".join(f"{c:>{max(len(c), 5)}s}" for c in COLUMNS))
    for name, row in summarize(results).items():
        print(f"{name:{width}s}  " + "  ".join(f"{fmt(row[c]):>{max(len(c), 5)}s}" for c in COLUMNS))


if __name__ == "__main__":
    main()
