"""Recall: grounded long-term memory for AI assistants and support teams. A Streamlit demo of our solution.

    streamlit run app.py                     # cached Reactor answers only: never calls Reactor
    REACTOR_LIVE=1 streamlit run app.py      # adds an "Ask Reactor" button (REACTOR_BASE_URL, REACTOR_AGENT_ID,
                                             # REACTOR_ACCESS_KEY from the environment; the key is never shown)

Answers come from the run files in the repo root: every results-*.jsonl with its <name>.evidence.jsonl, as written
by run.py / run_v2.py. Retrieval runs live for any question, cached or not. The Results tab reads metrics/*.json
(eval_retrieval.py --json) and scores the cached dev runs with score.py.
"""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from make_predictions import predict
from retrieval import HistoryIndex, is_aggregate
from score import score_question, summarize

ROOT = Path(__file__).resolve().parent
SPLITS = {"dev": ROOT / "dev", "test": ROOT / "test"}
LIVE = os.environ.get("REACTOR_LIVE") == "1"
LIVE_ENV = ("REACTOR_BASE_URL", "REACTOR_AGENT_ID", "REACTOR_ACCESS_KEY")
HISTORY_ID = re.compile(rb'\s*\{\s*"history_id"\s*:\s*"([^"]+)"')
NOTE_DAYS = re.compile(r"^\((\d+) days? before the question date\)$")
ROLE = re.compile(r"^(user \(elsewhere in this chat\)|user|assistant): ", re.MULTILINE)

st.set_page_config(page_title="Recall", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.card {background: #161b26; border: 1px solid #262d3d; border-radius: 12px; padding: 18px 22px; margin-bottom: 12px;}
.answer {font-size: 1.6rem; font-weight: 600; line-height: 1.35; color: #f2f4f8;}
.muted {color: #8a93a6; font-size: 0.85rem;}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
        margin-right: 6px; border: 1px solid transparent;}
.b-abstain {background: #3a2a12; color: #ffb547; border-color: #6b4a14;}
.b-ok {background: #11331f; color: #4cd787; border-color: #1d5c37;}
.b-bad {background: #3a1518; color: #ff6b74; border-color: #6b2329;}
.b-judge {background: #23283a; color: #a9b4d0; border-color: #39415a;}
.b-cited {background: #1b2a52; color: #9db4ff; border-color: #34508f;}
.b-gold {background: #33300f; color: #e9d76b; border-color: #5e5719;}
.b-recent {background: #12303a; color: #5fd0e8; border-color: #1f5563;}
.excerpt {background: #12161f; border: 1px solid #232a38; border-left: 4px solid #2a3142; border-radius: 10px;
          padding: 12px 16px; margin-bottom: 10px;}
.excerpt.cited {border-left-color: #7c9cff; background: #141b2c;}
.excerpt pre {white-space: pre-wrap; font-family: inherit; font-size: 0.86rem; color: #c9cfdb; margin: 8px 0 0 0;
              max-height: 260px; overflow-y: auto;}
.role {font-weight: 700; color: #7c9cff;}
.role-a {font-weight: 700; color: #b48cff;}
.hint {background: #2b2410; border: 1px solid #5e4c14; border-radius: 10px; padding: 10px 14px; color: #f0d58a;}
</style>
""", unsafe_allow_html=True)


# ---------- data ----------

@st.cache_resource(show_spinner="Indexing the histories file (once)...")
def history_offsets(split: str) -> dict[str, int]:
    """Byte offset of each history's line, found by streaming the file once without parsing it."""
    offsets, pos = {}, 0
    with (SPLITS[split] / "histories.jsonl").open("rb") as f:
        for line in f:
            m = HISTORY_ID.match(line[:200])
            if m:
                offsets[m.group(1).decode()] = pos
            pos += len(line)
    return offsets


@st.cache_resource(max_entries=12, show_spinner="Building the BM25 index for this history...")
def load_index(split: str, history_id: str) -> tuple[HistoryIndex, int, int]:
    """(index, the full history's characters, its session count), reading only that history's line."""
    with (SPLITS[split] / "histories.jsonl").open("rb") as f:
        f.seek(history_offsets(split)[history_id])
        history = json.loads(f.readline())
    chars = sum(len(s["session_id"]) + len(s["date"]) + 4 + sum(len(t["role"]) + 2 + len(t["content"]) + 1
                                                             for t in s["turns"]) for s in history["sessions"])
    return HistoryIndex(history), chars, len(history["sessions"])


def read_jsonl(path: Path) -> list[dict]:
    """JSON lines, skipping a line still being written by a running benchmark."""
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


@st.cache_data
def load_questions(split: str) -> list[dict]:
    return read_jsonl(SPLITS[split] / "questions.jsonl")


@st.cache_data
def load_gold(split: str) -> dict[str, dict]:
    return {g["question_id"]: g for g in read_jsonl(SPLITS[split] / "gold.jsonl")}


@st.cache_data(ttl=30)
def load_runs() -> dict[str, list[dict]]:
    """Cached Reactor answers by question_id: [{file, version, result, logged}] from every results-*.jsonl."""
    runs: dict[str, list[dict]] = {}
    for path in sorted(ROOT.glob("results*.jsonl")):
        if path.name.endswith(".evidence.jsonl"):
            continue
        results: dict[str, dict] = {}
        for r in read_jsonl(path):
            qid = r.get("question_id")
            if qid and (r.get("status") == "answered" or results.get(qid, {}).get("status") != "answered"):
                results[qid] = r
        logged = {e["question_id"]: e for e in read_jsonl(path.with_suffix(".evidence.jsonl")) if "question_id" in e}
        for qid, result in results.items():
            entry = logged.get(qid)
            runs.setdefault(qid, []).append({"file": path.name, "version": (entry or {}).get("version", 1),
                                             "result": result, "logged": entry})
    return runs


# ---------- helpers ----------

def date_of(value: str) -> datetime:
    return datetime.strptime(value[:10], "%Y/%m/%d")


def parse_excerpt(text: str) -> tuple[int | None, bool, str]:
    """(days before the question, [most recent], the history text) of one excerpt."""
    days, recent, lines = None, False, text.split("\n")
    while lines:
        m = NOTE_DAYS.match(lines[0])
        if m:
            days = int(m.group(1))
        elif lines[0] == "[most recent]":
            recent = True
        else:
            break
        lines.pop(0)
    return days, recent, "\n".join(lines)


def render_turns(text: str) -> str:
    """Excerpt text as HTML, with the speaker labels styled."""
    def role(m: re.Match) -> str:
        cls = "role-a" if m.group(1) == "assistant" else "role"
        label = "user · elsewhere in this chat" if m.group(1).startswith("user (") else m.group(1)
        return f'\x00{cls}\x01{label}\x02'
    marked = html.escape(ROLE.sub(role, text))
    return re.sub("\x00(.*?)\x01(.*?)\x02", r'<span class="\1">\2:</span> ', marked)


def badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def fmt_int(n: float) -> str:
    return f"{n:,.0f}"


def show_table(rows: dict[str, dict]) -> None:
    """A metrics table: counts as integers, rates to three decimals, missing values as a dash."""
    cell = lambda v: "–" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    st.dataframe(pd.DataFrame({k: {c: cell(v) for c, v in row.items()} for k, row in rows.items()}).T,
                 width="stretch")


def ask_reactor(split: str, history_id: str, qid: str, question: str, question_date: str, version: int,
                evidence: list[dict], missing: list[str]) -> dict:
    """One question through reactor/client.py (REACTOR_LIVE=1 only). One Reactor history per dataset history."""
    from run import MAX_CHARS, ROUTED_MAX_CHARS, client, prompt
    from run_v2 import prompt_v2
    args = SimpleNamespace(client=ROOT / "reactor" / "client.py", base_url=os.environ["REACTOR_BASE_URL"],
                           agent_id=os.environ["REACTOR_AGENT_ID"])
    histories = st.session_state.setdefault("reactor_histories", {})
    key = f"{split}:{history_id}"
    if key not in histories:
        new = client(args, "--new-history")
        if new.get("status") != "history":
            return {"status": "error", "error": f"--new-history failed: {new.get('error') or new.get('status')}"}
        histories[key] = new["history_id"]
    text = prompt(question) if version == 1 else prompt_v2(question, missing)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evidence.json"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        extra = ["--history", histories[key], "--question", text, "--question-date", question_date,
                 "--question-id", qid, "--max-chars", str(ROUTED_MAX_CHARS if is_aggregate(question) else MAX_CHARS)]
        if evidence:
            extra += ["--evidence", str(path)]
        return client(args, *extra)


# ---------- sidebar ----------

st.sidebar.markdown("## 🧠 Recall")
st.sidebar.caption("Grounded long-term memory for AI assistants and support teams")
split = st.sidebar.radio("Dataset", list(SPLITS), horizontal=True,
                         help="dev has gold answers; test is the scored split, with no gold released")
questions = load_questions(split)
gold = load_gold(split)
runs = load_runs()
cached_ids = {q["question_id"] for q in questions if q["question_id"] in runs}

only_cached = st.sidebar.toggle(f"Only questions with a cached answer ({len(cached_ids)})", value=bool(cached_ids))
pool = [q for q in questions if not only_cached or q["question_id"] in cached_ids] or questions


def label(q: dict) -> str:
    g = gold.get(q["question_id"])
    tag = ("unanswerable" if g["abstain"] else g["question_type"]) if g else ""
    mark = "● " if q["question_id"] in runs else "○ "
    return mark + (f"[{tag}] " if tag else "") + q["question"][:90]


chosen = st.sidebar.selectbox("Question", pool, format_func=label)
own = st.sidebar.toggle("Ask your own question about this history")
if own:
    question_text = st.sidebar.text_area("Your question", value="", placeholder="e.g. Where did I go on my last trip?")
    day = st.sidebar.date_input("Asked on", value=date_of(chosen["question_date"]).date())
    question_date = f"{day:%Y/%m/%d} ({day:%a}) {chosen['question_date'][-5:]}"
    qid = "q_custom"
else:
    question_text, question_date, qid = chosen["question"], chosen["question_date"], chosen["question_id"]

version = 1 if st.sidebar.radio("Retrieval", ["v1", "v2"], horizontal=True,
                                help="v1: what the test run sends. v2: focused trimming, a bigger share for a "
                                     "leading session, time sentences, the missing-words hint") == "v1" else 2

st.sidebar.divider()
if LIVE and all(os.environ.get(k) for k in LIVE_ENV):
    st.sidebar.markdown(badge("LIVE REACTOR: ON", "b-ok"), unsafe_allow_html=True)
    st.sidebar.caption("Asking starts a new Reactor history on this agent, which interrupts a benchmark running "
                       "on it.")
elif LIVE:
    st.sidebar.markdown(badge("LIVE REACTOR: NOT CONFIGURED", "b-bad"), unsafe_allow_html=True)
    st.sidebar.caption("Set " + ", ".join(k for k in LIVE_ENV if not os.environ.get(k)) + ".")
else:
    st.sidebar.markdown(badge("LIVE REACTOR: OFF", "b-judge"), unsafe_allow_html=True)
    st.sidebar.caption("Cached answers only. Start with REACTOR_LIVE=1 to ask Reactor.")
live_ready = LIVE and all(os.environ.get(k) for k in LIVE_ENV)

ask_tab, results_tab = st.tabs(["Ask", "Results"])

# ---------- Ask ----------

def render_ask() -> None:
    if not question_text.strip():
        st.info("Type a question in the sidebar.")
        return
    index, history_chars, n_sessions = load_index(split, chosen["history_id"])
    run = None if own else next((r for r in reversed(runs.get(qid, [])) if r["version"] == version), None)
    live_key = (split, chosen["history_id"], question_text, question_date, version)
    live_result = st.session_state.get("live", {}).get(live_key)

    if run and run["logged"]:
        evidence, source_note = run["logged"]["evidence"], f"exactly what was sent ({run['file']})"
        missing = run["logged"].get("missing") or (index.missing_words(question_text, evidence) if version == 2 else [])
    else:
        evidence = index.evidence(question_text, question_date=question_date, version=version)
        missing = index.missing_words(question_text, evidence) if version == 2 else []
        source_note = f"live retrieval, v{version}"
    g = None if own else gold.get(qid)

    st.markdown(f"### {html.escape(question_text)}")
    st.caption(f"Asked {question_date} · history {chosen['history_id']} · {n_sessions} sessions · {split}"
               + (f" · {g['question_type']}" if g else ""))

    result = live_result or (run["result"] if run else None)
    left, right = st.columns([3, 2])
    with left:
        if result:
            prediction, cite_source = predict(qid, result, {"evidence": evidence})
            abstained = prediction["abstain"]
            status = result.get("status")
            if status != "answered":
                body = f"No answer: {html.escape(status or 'error')}"
                badges = badge("ABSTAINED", "b-abstain")
            elif abstained:
                body, badges = "I don't know: the history doesn't contain this.", badge("ABSTAINED", "b-abstain")
            else:
                body, badges = html.escape(prediction["answer"]), ""
            meta = []
            if result.get("latency_s") is not None:
                meta.append(f"{result['latency_s']:.0f} s")
            meta.append("live" if live_result else f"cached · {run['file']}")
            if not abstained:
                meta.append(f"citations from {cite_source}")
            st.markdown(f'<div class="card"><div class="muted">Reactor answer</div>{badges}'
                        f'<div class="answer">{body}</div><div class="muted">{" · ".join(meta)}</div></div>',
                        unsafe_allow_html=True)
            cited = set(prediction["citations"])
        else:
            prediction, cited = None, set()
            st.markdown('<div class="card"><div class="muted">Reactor answer</div><div class="answer" '
                        'style="font-size:1.1rem;color:#8a93a6">Reactor answer not cached'
                        f'{" for v" + str(version) if runs.get(qid) else ""}</div>'
                        '<div class="muted">Retrieval below is live; no Reactor call was made.</div></div>',
                        unsafe_allow_html=True)
            if live_ready and st.button("Ask Reactor", type="primary"):
                with st.spinner("Reactor is answering (typically under a minute)..."):
                    answer = ask_reactor(split, chosen["history_id"], qid, question_text, question_date, version,
                                         evidence, missing)
                st.session_state.setdefault("live", {})[live_key] = answer
                st.rerun()
    with right:
        if g:
            if prediction:
                verdict = score_question(prediction, g)
                mark = {"correct": badge("CORRECT", "b-ok"), "wrong": badge("INCORRECT", "b-bad"),
                        "unjudged": badge("NEEDS JUDGE", "b-judge")}[verdict["answer"]]
                f1 = verdict["evidence_f1"]
                extra = f'<div class="muted">evidence F1 {f1:.2f}</div>' if f1 is not None else ""
            else:
                mark, extra = "", ""
            expected = "Unanswerable: " + g["answer"] if g["abstain"] else g["answer"]
            st.markdown(f'<div class="card"><div class="muted">Gold answer</div>{mark}'
                        f'<div style="font-size:1.05rem;margin-top:6px">{html.escape(expected)}</div>{extra}'
                        f'<div class="muted" style="margin-top:6px">"Needs judge": the answer is not an exact '
                        f'match; official scoring sends it to an LLM judge.</div></div>', unsafe_allow_html=True)
        elif split == "test":
            st.markdown('<div class="card"><div class="muted">Gold answer</div>Not released for the test split.'
                        '</div>', unsafe_allow_html=True)

    # efficiency
    sent_chars = sum(len(e["text"]) for e in evidence) + len(question_text)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters sent", fmt_int(sent_chars), help="evidence excerpts + question")
    c2.metric("Full history", fmt_int(history_chars), help="every turn of every session in this history")
    c3.metric("Reduction", f"{history_chars / max(sent_chars, 1):.0f}×")
    c4.metric("Tokens (≈ chars/4)", f"{fmt_int(sent_chars / 4)} vs {fmt_int(history_chars / 4)}")

    # evidence timeline
    st.markdown("#### Evidence timeline")
    st.caption(f"{len(evidence)} excerpts in date order · {source_note}. Highlighted: cited for the answer.")
    if missing:
        st.markdown(f'<div class="hint">⚠ Missing-words hint (v2): the evidence never mentions '
                    f'<b>{html.escape(", ".join(missing))}</b>. Reactor is told to abstain if the answer depends on '
                    f'them: this is how near-miss traps are caught.</div>', unsafe_allow_html=True)
        st.write("")
    gold_sessions = set(g["evidence"]) if g else set()
    for n, e in enumerate(evidence, 1):
        days, recent, body = parse_excerpt(e["text"])
        chips = [badge(f"E{n}", "b-judge")]
        if e["id"] in cited:
            chips.append(badge("cited", "b-cited"))
        if recent:
            chips.append(badge("most recent", "b-recent"))
        if e["id"] in gold_sessions:
            chips.append(badge("gold evidence", "b-gold"))
        when = e.get("date", "")
        if days is not None:
            when += f" · {days} day{'' if days == 1 else 's'} before the question"
        st.markdown(f'<div class="excerpt{" cited" if e["id"] in cited else ""}">{"".join(chips)}'
                    f'<span class="muted">{html.escape(when)} · {html.escape(e["id"])}</span>'
                    f'<pre>{render_turns(body)}</pre></div>', unsafe_allow_html=True)
    if not evidence:
        st.info("Nothing in this history matches the question's words.")

    with st.expander("How it works"):
        st.markdown(f"""
**Recall** answers questions about months of past conversations: it finds the few relevant turns, has Reactor
answer from them with citations, and says *I don't know* when the history doesn't hold the answer.

1. **Index.** Every turn of every session, user and assistant, becomes a BM25 chunk (long turns split into
   1,500-character windows). Built once per history, in under a second, standard library only.
2. **Route.** Questions that count, total, compare or order ("how many", "which first", "how long ago") get
   8 sessions and 24,000 characters; others get 5 sessions and 12,000.
3. **Retrieve and trim.** Sessions are ranked by their best turns. Each excerpt keeps the best-matching turns
   with one turn either side, its date and how many days before the question it was. v2 trims long turns around
   the question's words, gives a clearly leading session more room, and adds the user's own time references
   ("last month", "three weeks ago") for temporal questions.
4. **Reactor answers with citations.** The question goes to Reactor with the excerpts labelled [E1], [E2]...
   and instructions: cite on the answer line, one value only, the most recent value wins, and work out
   counts, totals and orders step by step.
5. **Support check / abstain.** Citations come from Reactor's labels; without labels, only from excerpts that
   actually contain the answer. If Reactor says INSUFFICIENT_EVIDENCE, Recall abstains, with no citations.

**Edge cases handled**
- **Knowledge updates**: excerpts are in date order and the latest is marked *most recent*; Reactor is told the
  most recent value wins, except when the question asks for the previous one.
- **Temporal questions**: each excerpt says how many days before the question it happened, computed in code.
- **Near-miss traps**: v2 names the question's words that appear nowhere in the evidence (the *missing-words
  hint*) so that Reactor abstains instead of stitching an answer together.
- **Assistant-said facts**: assistant turns are indexed too, so "what did you recommend" questions find them.
- **Unanswerable questions**: Reactor's INSUFFICIENT_EVIDENCE becomes an abstention with no citations.

**Why it matters**: this history is {fmt_int(history_chars)} characters; Recall sends {fmt_int(sent_chars)}
({history_chars / max(sent_chars, 1):.0f}× less), so each answer is cheaper and faster, and every claim
points to the conversation it came from.
""")


with ask_tab:
    render_ask()

# ---------- Results ----------

with results_tab:
    dev_gold = load_gold("dev")
    files = sorted({r["file"] for rs in runs.values() for r in rs})
    st.markdown("#### Answer accuracy on cached dev runs")
    shown = False
    for name in files:
        entries = [(qid, r) for qid, rs in runs.items() for r in rs if r["file"] == name and qid in dev_gold]
        if not entries:
            continue
        shown = True
        scored = [score_question(predict(qid, r["result"], r["logged"])[0], dev_gold[qid]) for qid, r in entries]
        st.markdown(f"**{name}** · {len(entries)} questions · retrieval v{entries[0][1]['version']}")
        show_table(summarize(scored))
    if not shown:
        st.info("No cached dev runs yet (results-*.jsonl in the repo root).")
    st.caption("score.py's automatic matcher: 'unjudged' answers go to the LLM judge in official scoring.")

    test_runs = [r for q in load_questions("test") for r in runs.get(q["question_id"], [])]
    if test_runs:
        answered = sum(r["result"].get("status") == "answered" for r in test_runs)
        abstained = sum(bool(r["result"].get("abstained")) for r in test_runs)
        st.markdown("#### Test run")
        st.write(f"{len(test_runs)} of {len(load_questions('test'))} test questions have a result "
                 f"({answered} answered, {abstained} abstained). No gold is released for test.")

    st.markdown("#### Retrieval quality (offline, no Reactor)")
    st.caption("hit@5: a gold session is among the top 5 · sent: share of gold sessions in the evidence sent · "
               "ans∈ev: the gold answer appears in the evidence sent. From eval_retrieval.py --json.")
    for split_name, title in (("dev", "dev (all 526 questions, ~50 sessions per history)"),
                              ("dev-large-130", "dev-large (130 questions, ~500 sessions per history)")):
        tables = {v: ROOT / "metrics" / f"retrieval-{split_name}-v{v}.json" for v in (1, 2)}
        if not all(p.exists() for p in tables.values()):
            continue
        data = {v: json.loads(p.read_text(encoding="utf-8"))["rows"] for v, p in tables.items()}
        rows = {}
        for key in data[1]:
            rows[key] = {"n": data[1][key]["n"]}
            for metric in ("hit@5", "sent", "ans∈ev"):
                for v in (1, 2):
                    rows[key][f"{metric} v{v}"] = data[v][key].get(metric)
        st.markdown(f"**{title}**")
        show_table(rows)
