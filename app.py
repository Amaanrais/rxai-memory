"""Recall: find answers in months of chat history, with sources. A Streamlit demo of our solution.

    streamlit run app.py --server.address localhost          # saved Reactor answers only: never calls Reactor
    REACTOR_LIVE=1 streamlit run app.py --server.address localhost
        # adds an "Ask Reactor" button (REACTOR_BASE_URL, REACTOR_AGENT_ID, REACTOR_ACCESS_KEY from the environment;
        # the key is never shown)

Answers come from the run files in the repo root: every results-*.jsonl with its <name>.evidence.jsonl, as written
by run.py / run_v2.py. Retrieval runs live for any question, answered or not. The Results tab reads metrics/*.json
(eval_retrieval.py --json) and scores the saved dev runs with score.py.
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

import streamlit as st

from make_predictions import choose, predict
from retrieval import STOPWORDS, HistoryIndex, is_aggregate, stem, tokenize
from score import score_question, summarize

ROOT = Path(__file__).resolve().parent
SPLITS = {"dev": ROOT / "dev", "test": ROOT / "test"}
LIVE = os.environ.get("REACTOR_LIVE") == "1"
LIVE_ENV = ("REACTOR_BASE_URL", "REACTOR_AGENT_ID", "REACTOR_ACCESS_KEY")
HISTORY_ID = re.compile(rb'\s*\{\s*"history_id"\s*:\s*"([^"]+)"')
NOTE_DAYS = re.compile(r"^\((\d+) days? before the question date\)$")
TURN = re.compile(r"^(user \(elsewhere in this chat\)|user|assistant): ", re.MULTILINE)
WORD = re.compile(r"[A-Za-z0-9]+")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# the four questions of the "Try these" row: dev questions with a saved answer, one per kind of question
DEMOS = [("A fact the user mentioned", "q_4feb114c6944"),
         ("A fact that changed", "q_92447a8c097c"),
         ("A trap: the answer isn't there", "q_43824099a143"),
         ("Something the assistant said", "q_14252c64d73d")]

st.set_page_config(page_title="Recall", page_icon="🔎", layout="centered", initial_sidebar_state="collapsed")

# ---------- style ----------
# Framed sheet (taste-skill app style, seed 655928746): the page ground #f2f4f6, one white sheet with a 1px edge and
# 16px corners, no shadows. One accent, teal, on the mark, the active tab, the efficiency bar, links and focus. The
# highlighter is only ever on cited text.
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');
:root {
  --ground: #f2f4f6; --surface: #ffffff; --raised: #f7f8fa; --active: #e9edf1;
  --border: #e3e7eb; --rule: #edf0f3; --control: #cbd2d9;
  --ink: #1c2430; --muted: #5e6876; --accent: #0e5a5f; --accent-soft: #e3eeee;
  --mark: rgba(255, 230, 138, 0.6); --notice-bg: #f6f2e3; --notice-ink: #5c4a12;
  --font: "Schibsted Grotesk", system-ui, sans-serif;
}
html, body, .stApp, .stApp * { font-family: var(--font); }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
.stApp { background: var(--ground); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
.stDeployButton, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container { max-width: 808px; padding: 40px 24px 96px; }
[data-testid="stSidebar"] { background: var(--ground); border-right: 1px solid var(--border); }
*:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* header */
.brand { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin: 0 0 4px; }
.brand .word { font-size: 22px; font-weight: 700; letter-spacing: -0.01em; color: var(--ink); }
.brand .word b { color: var(--accent); font-weight: 700; }
.brand .line { font-size: 15px; color: var(--muted); }
.try { font-size: 14px; color: var(--muted); margin: 22px 0 8px; }

/* buttons: quiet, white, 1px control edge; the chosen demo takes the neutral active fill */
.stButton > button { background: var(--surface); color: var(--ink); border: 1px solid var(--control);
  border-radius: 8px; min-height: 44px; font-size: 14px; font-weight: 500; line-height: 1.3;
  padding: 8px 12px; box-shadow: none; text-align: left; justify-content: flex-start; }
.stButton > button > div, .stButton > button > div > span { justify-content: flex-start; width: 100%; }
.stButton > button:hover { border-color: var(--ink); color: var(--ink); background: var(--surface); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] { background: var(--active);
  border-color: var(--ink); color: var(--ink); }
.stButton > button p, .stButton > button div { font-size: 14px; white-space: normal; overflow: visible;
  text-overflow: clip; text-align: left; }
[data-testid="stExpander"] details { border: none; background: transparent; }
[data-testid="stExpander"] summary { padding-left: 0; color: var(--muted); font-size: 14px; }
[data-testid="stExpander"] summary:hover { color: var(--ink); }

/* the sheet */
.st-key-sheet { background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 32px 32px 24px; margin-top: 12px; }
h1.question { font-size: 30px; line-height: 1.2; font-weight: 600; letter-spacing: -0.01em; color: var(--ink);
  margin: 0; padding: 0; }
.meta { font-size: 15px; color: var(--muted); margin-top: 8px; }
.answer-label { font-size: 14px; color: var(--muted); margin-top: 28px; }
.answer { font-size: 40px; line-height: 1.15; font-weight: 600; letter-spacing: -0.02em; color: var(--ink);
  margin-top: 6px; overflow-wrap: anywhere; }
.answer.long { font-size: 28px; line-height: 1.25; }
.answer.none { font-size: 24px; font-weight: 500; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.chip { font-size: 13px; color: var(--accent); background: var(--accent-soft); border-radius: 999px;
  padding: 4px 11px; text-decoration: none; white-space: nowrap; }
.chip:hover { text-decoration: underline; }
.status { font-size: 16px; color: var(--ink); margin-top: 16px; }
.status .why { color: var(--muted); }
.efficiency { margin-top: 26px; padding-top: 18px; border-top: 1px solid var(--rule); font-size: 15px;
  color: var(--muted); }
.efficiency b { color: var(--ink); font-weight: 600; }
.bar { height: 4px; background: var(--rule); border-radius: 2px; margin-top: 10px; overflow: hidden; }
.bar span { display: block; height: 4px; background: var(--accent); border-radius: 2px; min-width: 3px; }

/* tabs: a 2px teal underline on a 1px rule */
.stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--border); margin-top: 20px; }
.stTabs [data-baseweb="tab"] { padding: 10px 0; font-size: 15px; color: var(--muted); }
.stTabs [data-baseweb="tab"] p { font-size: 15px; }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* sources */
.sources-intro { font-size: 14px; color: var(--muted); margin: 4px 0 12px; }
.hint { background: var(--notice-bg); color: var(--notice-ink); border-radius: 10px; padding: 12px 14px;
  font-size: 15px; line-height: 1.5; margin-bottom: 12px; }
details.src { border-top: 1px solid var(--rule); padding: 14px 0; }
details.src:last-of-type { border-bottom: 1px solid var(--rule); }
details.src summary { list-style: none; cursor: pointer; border-radius: 6px; }
details.src summary::-webkit-details-marker { display: none; }
.src .when { font-size: 15px; font-weight: 600; color: var(--ink); }
.src .when .ago { font-weight: 400; color: var(--muted); }
.src .when .tag { font-weight: 500; color: var(--accent); margin-left: 6px; }
.src .snippet { font-size: 15px; line-height: 1.5; color: var(--ink); margin-top: 4px; }
.src .snippet .open { color: var(--muted); font-size: 14px; margin-left: 6px; white-space: nowrap; }
details.src[open] .open { display: none; }
.src .turns { margin-top: 14px; padding: 16px 18px; background: var(--raised); border-radius: 10px;
  max-height: 460px; overflow-y: auto; }
.src .turn { margin-bottom: 14px; }
.src .turn:last-child { margin-bottom: 0; }
.src .who { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 2px; }
.src .said { font-size: 15px; line-height: 1.55; color: var(--ink); white-space: pre-wrap; overflow-wrap: anywhere; }
.src .gap { color: var(--muted); font-size: 14px; margin-bottom: 14px; }
mark.hl { background: var(--mark); color: inherit; padding: 0 2px; border-radius: 2px;
  -webkit-box-decoration-break: clone; box-decoration-break: clone; }

/* how it works, results */
.prose { font-size: 16px; line-height: 1.6; color: var(--ink); }
.prose h3 { font-size: 18px; font-weight: 600; margin: 22px 0 6px; }
.prose ol, .prose ul { padding-left: 22px; margin: 0; }
.prose li { margin-bottom: 8px; }
.prose li::marker { font-variant-numeric: normal; }
.prose .muted { color: var(--muted); }
table.t { border-collapse: collapse; width: 100%; font-size: 14px; margin: 6px 0 4px; }
table.t td { font-variant-numeric: tabular-nums; }
table.t td:first-child { font-variant-numeric: normal; }
table.t, table.t th, table.t td { font-family: var(--font) !important; }
table.t th { font-weight: 400; color: var(--muted); text-align: right; padding: 6px 8px;
  border-bottom: 1px solid var(--border); white-space: nowrap; }
table.t td { padding: 7px 8px; text-align: right; border-bottom: 1px solid var(--rule); }
table.t th:first-child, table.t td:first-child { text-align: left; padding-left: 0; width: 34%; }
table.t tr.all td { font-weight: 600; }
.tcap { font-size: 13px; color: var(--muted); margin-bottom: 18px; }

/* sidebar details */
.adv { margin: 0; padding: 0; }
.adv dt { font-size: 13px; color: var(--muted); margin-top: 10px; }
.adv dd { font-size: 14px; color: var(--ink); margin: 2px 0 0; overflow-wrap: anywhere; }
@media (max-width: 640px) {
  .block-container { padding: 64px 16px 64px; }
  .st-key-sheet { padding: 22px 18px 18px; border-radius: 12px; }
  h1.question { font-size: 24px; }
  .answer { font-size: 30px; }
  .answer.long { font-size: 22px; }
}
</style>
""")


# ---------- data ----------

@st.cache_resource(show_spinner="Reading the histories file once...")
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


@st.cache_resource(max_entries=12, show_spinner="Indexing this history...")
def load_index(split: str, history_id: str) -> tuple[HistoryIndex, int, int]:
    """(index, the full history's characters, its conversation count), reading only that history's line."""
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
    """Saved Reactor answers by question_id: [{file, version, result, logged}], one per results-*.jsonl, the line
    make_predictions.py would use."""
    runs: dict[str, list[dict]] = {}
    for path in sorted(ROOT.glob("results*.jsonl")):
        if path.name.endswith(".evidence.jsonl"):
            continue
        logged = {e["question_id"]: e for e in read_jsonl(path.with_suffix(".evidence.jsonl")) if "question_id" in e}
        for qid, result in choose(read_jsonl(path)).items():
            entry = logged.get(qid)
            runs.setdefault(qid, []).append({"file": path.name, "version": (entry or {}).get("version", 1),
                                             "result": result, "logged": entry})
    return runs


# ---------- text helpers ----------

def date_of(value: str) -> datetime:
    return datetime.strptime(value[:10], "%Y/%m/%d")


def day(value: str, year: bool = True) -> str:
    d = date_of(value)
    return f"{d.day} {d:%B %Y}" if year else f"{d.day} {d:%b}"


def short(n: float) -> str:
    """19,591 -> 19.6k; 497,191 -> 497k."""
    if n < 1000:
        return f"{n:,.0f}"
    k = n / 1000
    return f"{k:,.0f}k" if k >= 100 else f"{k:.1f}k"


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


MARKUP = re.compile(r"\*\*|__|^#{1,6}\s+", re.MULTILINE)


def plain(text: str) -> str:
    """Chat text as plain text: markdown's bold and heading markers dropped, nothing else interpreted."""
    return MARKUP.sub("", text)


def turns_of(body: str) -> list[tuple[str, str]]:
    """[(speaker, text)] of an excerpt's history text; ("", "...") marks a gap between turns."""
    out, last = [], 0
    marks = list(TURN.finditer(body))
    for n, m in enumerate(marks):
        text = body[m.end():marks[n + 1].start() if n + 1 < len(marks) else len(body)].strip("\n")
        who = {"user": "You", "assistant": "Assistant"}.get(m.group(1), "You, elsewhere in this conversation")
        gap = text.endswith("\n...")
        out.append((who, plain(text[:-4] if gap else text)))
        if gap:
            out.append(("", "..."))
    return out


def keywords(question: str) -> set[str]:
    """The question's stemmed content words."""
    return set(tokenize(question))


def best_sentence(text: str, words: set[str]) -> str:
    """The sentence of ``text`` that holds the most question words (the first sentence when none does)."""
    best, score = "", -1
    for sentence in SENTENCE.split(text):
        sentence = sentence.strip()
        if len(sentence) < 3:
            continue
        hits = len({stem(w.lower()) for w in WORD.findall(sentence) if w.lower() not in STOPWORDS} & words)
        if hits > score:
            best, score = sentence, hits
    return best


def clip_line(text: str, limit: int = 150) -> str:
    """One line of ``text``, cut at a word; "…" marks text cut at either end (retrieval trims long turns)."""
    text = " ".join(text.split())
    if text[:1].islower():
        text = "…" + text
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def highlighted(text: str, sentence: str) -> str:
    """``text`` escaped, with ``sentence`` (if present) under the highlighter."""
    i = text.find(sentence) if sentence else -1
    if i < 0:
        return html.escape(text)
    return (html.escape(text[:i]) + f'<mark class="hl">{html.escape(sentence)}</mark>'
            + html.escape(text[i + len(sentence):]))


def table(rows: dict[str, dict], columns: list[tuple[str, str]], fmt=None) -> str:
    """An HTML table: the row name left, numbers right-aligned in tabular figures, a dash for a missing value."""
    cell = fmt or (lambda v: "–" if v is None else f"{v:.1%}" if isinstance(v, float) else str(v))
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = ""
    for name, row in rows.items():
        cls = ' class="all"' if name.lower() in ("all", "overall") else ""
        body += f"<tr{cls}><td>{html.escape(name)}</td>" + "".join(
            f"<td>{cell(row.get(key))}</td>" for key, _ in columns) + "</tr>"
    return f'<table class="t"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>'


TYPE_NAMES = {"knowledge-update": "A fact that changed", "multi-session": "Across conversations",
              "single-session-assistant": "Something the assistant said",
              "single-session-user": "A fact the user mentioned", "temporal-reasoning": "Dates and order",
              "all": "All"}


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


# ---------- sidebar: Advanced ----------

st.sidebar.markdown("### Advanced")
split = st.sidebar.radio("Dataset", list(SPLITS), horizontal=True,
                         help="dev has expected answers; test is the scored set, with none released")
version = 1 if st.sidebar.radio("Retrieval version", ["v1", "v2"], horizontal=True,
                                help="v1 produced the submitted answers. v2 adds focused trimming, more room for a "
                                     "leading conversation, time sentences and the missing-words note.") == "v1" else 2
live_ready = LIVE and all(os.environ.get(k) for k in LIVE_ENV)
details_slot = st.sidebar.container()
st.sidebar.markdown("#### Live mode")
if live_ready:
    st.sidebar.write("On. Asking starts a new Reactor history on this agent, which interrupts a benchmark running "
                     "on it.")
elif LIVE:
    st.sidebar.write("Not configured: set " + ", ".join(k for k in LIVE_ENV if not os.environ.get(k)) + ".")
else:
    st.sidebar.write("Off. Answers shown are the ones saved from our runs. Start with REACTOR_LIVE=1 to ask Reactor.")

questions = load_questions(split)
by_id = {q["question_id"]: q for q in questions}
gold = load_gold(split)
runs = load_runs()

# ---------- header and question picker ----------

st.html('<div class="brand"><span class="word">Recall<b>.</b></span>'
        '<span class="line">Find answers in months of chat history, with sources.</span></div>')

state = st.session_state
if state.get("split") != split:  # a new dataset starts on its first demo, or its first answered question
    state.split = split
    first = next((qid for _, qid in DEMOS if qid in by_id), None)
    state.qid = first or next((q["question_id"] for q in questions if q["question_id"] in runs),
                              questions[0]["question_id"])
    state.own = False

def show_question(qid: str) -> None:
    state.qid, state.own = qid, False


def ask_own() -> None:
    state.own, state.own_text, state.own_day = True, state.own_input.strip(), state.own_date


demos = [(name, qid) for name, qid in DEMOS if qid in by_id]
if demos:
    st.html('<div class="try">Try these</div>')
    cols = st.columns(len(demos))
    for col, (name, qid) in zip(cols, demos):
        chosen = not state.own and state.qid == qid
        col.button(name, key=f"demo-{qid}", type="primary" if chosen else "secondary", width="stretch",
                   on_click=show_question, args=(qid,))

with st.expander("Browse all questions or ask your own"):
    answered_only = st.toggle("Only questions with a saved answer", value=True)
    pool = [q for q in questions if not answered_only or q["question_id"] in runs] or questions
    ids = [q["question_id"] for q in pool]
    if state.qid in ids:
        state.browse = state.qid  # the list follows the question on screen, whichever control chose it
    elif state.get("browse") not in ids:
        state.browse = ids[0]
    st.selectbox("Question", ids, key="browse", format_func=lambda qid: by_id[qid]["question"],
                 on_change=lambda: show_question(state.browse))
    st.write("")
    st.text_input("Ask your own question about the same history", key="own_input",
                  placeholder="e.g. Where did I go on my last trip?")
    st.date_input("As if asked on", key="own_date", value=date_of(by_id[state.qid]["question_date"]).date())
    st.button("Ask", disabled=not state.get("own_input", "").strip(), on_click=ask_own)

base = by_id[state.qid]
if state.own:
    question_text = state.own_text
    d = state.own_day
    question_date = f"{d:%Y/%m/%d} ({d:%a}) {base['question_date'][-5:]}"
    qid = "q_custom"
else:
    question_text, question_date, qid = base["question"], base["question_date"], base["question_id"]

index, history_chars, n_sessions = load_index(split, base["history_id"])
run = None if state.own else next((r for r in reversed(runs.get(qid, [])) if r["version"] == version), None)
live_key = (split, base["history_id"], question_text, question_date, version)
live_result = state.get("live", {}).get(live_key)
if run and run["logged"]:
    evidence = run["logged"]["evidence"]
    missing = run["logged"].get("missing") or (index.missing_words(question_text, evidence) if version == 2 else [])
else:
    evidence = index.evidence(question_text, question_date=question_date, version=version)
    missing = index.missing_words(question_text, evidence) if version == 2 else []
g = None if state.own else gold.get(qid)
result = live_result or (run["result"] if run else None)
prediction = predict(qid, result, {"evidence": evidence})[0] if result else None
cite_source = predict(qid, result, {"evidence": evidence})[1] if result else "-"
cited = set(prediction["citations"]) if prediction else set()
dates = {e["id"]: e["date"] for e in evidence}
sent_chars = sum(len(e["text"]) for e in evidence) + len(question_text)
words = keywords(question_text)

# ---------- the sheet ----------

with st.container(key="sheet"):
    st.html(f'<h1 class="question">{html.escape(question_text)}</h1>'
            f'<div class="meta">Asked {day(question_date)}, searched {n_sessions} conversations</div>')

    said = []  # the status, in plain words
    if prediction is None:
        answer_html = ('<div class="answer none">Reactor hasn\'t been asked this one yet. '
                       'The sources Recall would send are below.</div>')
    elif result.get("status") != "answered":
        answer_html = '<div class="answer none">Reactor didn\'t answer in time.</div>'
        said.append('Recall treats this as "I don\'t know", with no sources.')
    elif prediction["abstain"]:
        answer_html = '<div class="answer">I don\'t know.</div>'
        said.append("Not in the history, so Recall says it doesn\'t know.")
    else:
        text = prediction["answer"]
        answer_html = f'<div class="answer{" long" if len(text) > 60 else ""}">{html.escape(text)}</div>'
    if prediction is not None and g:
        verdict = score_question(prediction, g)["answer"]
        expected = html.escape("It isn't in the history." if g["abstain"] else g["answer"])
        said.append({"correct": "Matches the expected answer.",
                     "unjudged": f'Checked by the judge later. <span class="why">Expected: {expected}</span>',
                     "wrong": f'Doesn\'t match. <span class="why">Expected: {expected}</span>'}[verdict])
    status = f'<div class="status">{" ".join(said)}</div>' if said else ""
    chips = "".join(f'<a class="chip" href="#src-{sid}">{day(dates[sid], year=False)} conversation</a>'
                    for sid in sorted(cited, key=lambda s: date_of(dates[s])) if sid in dates)
    ratio = history_chars / max(sent_chars, 1)
    st.html(f'<div class="answer-label">Answer</div>{answer_html}'
            + (f'<div class="chips">{chips}</div>' if chips else "") + status
            + f'<div class="efficiency">Read <b>{short(sent_chars)}</b> of {short(history_chars)} characters, '
              f'{ratio:.0f}× less than the full history.'
              f'<div class="bar" role="img" aria-label="{sent_chars / history_chars:.1%} of the history read">'
              f'<span style="width:{min(sent_chars / history_chars, 1):.2%}"></span></div></div>')

    if live_ready and prediction is None and st.button("Ask Reactor"):
        with st.spinner("Reactor is answering, usually within a minute..."):
            answer = ask_reactor(split, base["history_id"], qid, question_text, question_date, version,
                                 evidence, missing)
        state.setdefault("live", {})[live_key] = answer
        st.rerun()

    sources_tab, how_tab, results_tab = st.tabs(["Sources", "How it works", "Results"])

    with sources_tab:
        intro = (f"{len(evidence)} conversations Recall sent to Reactor, oldest first."
                 + (" Highlighted: the ones the answer cites." if cited else ""))
        parts = [f'<div class="sources-intro">{intro}</div>']
        if missing:
            parts.append(f'<div class="hint">These words from the question appear nowhere in these conversations: '
                         f'<b>{html.escape(", ".join(missing))}</b>. If the answer depends on them, Reactor is told '
                         f'to say it doesn\'t know. This is how near-miss traps are caught.</div>')
        for e in evidence:
            days, recent, body = parse_excerpt(e["text"])
            is_cited = e["id"] in cited
            turns = turns_of(body)
            spoken = " ".join(t for who, t in turns if who)
            snippet = best_sentence(spoken, words)
            ago = ("on the day of the question" if days == 0 else f"{days} day{'' if days == 1 else 's'} "
                   "before the question") if days is not None else ""
            tags = (' <span class="tag">cited</span>' if is_cited else "") + \
                   (' <span class="tag">most recent</span>' if recent else "")
            one_line = clip_line(snippet)
            shown = f'<mark class="hl">{html.escape(one_line)}</mark>' if is_cited else html.escape(one_line)
            inner = ""
            for who, text in turns:
                if not who:
                    inner += '<div class="gap">…</div>'
                    continue
                said = highlighted(text, snippet) if is_cited else html.escape(text)
                inner += f'<div class="turn"><div class="who">{html.escape(who)}:</div><div class="said">{said}</div></div>'
            parts.append(f'<details class="src" id="src-{html.escape(e["id"])}"><summary>'
                         f'<div class="when">{day(e["date"])}{", " + ago if ago else ""}'
                         f'{tags}</div>'
                         f'<div class="snippet">{shown}<span class="open">Show conversation</span></div>'
                         f'</summary><div class="turns">{inner}</div></details>')
        if not evidence:
            parts.append('<div class="sources-intro">Nothing in this history matches the question\'s words.</div>')
        st.html("".join(parts))

    with how_tab:
        st.html(f"""<div class="prose">
<h3>What Recall does</h3>
<ol>
<li><b>Reads the whole history once.</b> Every message, yours and the assistant's, is indexed so it can be
searched in a fraction of a second.</li>
<li><b>Works out what kind of question it is.</b> Questions that count, add up or put things in order get more
conversations and more room than a simple lookup.</li>
<li><b>Picks the few passages that matter.</b> From each matching conversation it keeps the best messages and the
ones either side, with the date and how long before the question it was.</li>
<li><b>Reactor answers from those passages.</b> It must point to the passages it used, give one value, and use the
most recent one when something changed.</li>
<li><b>Recall checks the sources, or says it doesn't know.</b> It cites only conversations that support the answer.
If Reactor finds nothing that answers the question, Recall says so, with no sources.</li>
</ol>
<h3>The hard cases it handles</h3>
<ul>
<li><b>Facts that changed:</b> passages are in date order and the latest is marked, so the newest value wins unless
you ask for the old one.</li>
<li><b>Dates and order:</b> each passage says how many days before the question it was, worked out in code.</li>
<li><b>Near-miss traps:</b> a question can be almost answerable. Recall names the words that appear nowhere in the
passages, so Reactor says it doesn't know instead of guessing.</li>
<li><b>Something the assistant said:</b> the assistant's messages are searched too, for "what did you recommend"
questions.</li>
<li><b>Questions with no answer:</b> "I don't know" is an answer, and it comes with no sources.</li>
</ul>
<p class="muted">On this history Recall read {short(sent_chars)} of {short(history_chars)} characters. Each answer
costs less, comes back faster, and points to the conversation it came from.</p>
</div>""")

    with results_tab:
        dev_gold = load_gold("dev")
        out = []
        files = sorted({r["file"] for rs in runs.values() for r in rs})
        for name in files:
            entries = [(q, r) for q, rs in runs.items() for r in rs if r["file"] == name and q in dev_gold]
            if not entries:
                continue
            scored = [score_question(predict(q, r["result"], r["logged"])[0], dev_gold[q]) for q, r in entries]
            rows = {TYPE_NAMES.get(k, k): v for k, v in summarize(scored).items()}
            out.append(f'<div class="prose"><h3>Answers on {len(entries)} practice questions</h3></div>')
            out.append(table(rows, [("n", "Questions"), ("correct", "Match"), ("unjudged", "For the judge"),
                                    ("abstention_recall", "Rightly said “don't know”"),
                                    ("evidence_f1", "Sources right")]))
            out.append('<div class="tcap">Scored with the organisers\' checker on one saved run. "For the judge": not '
                       'an exact match, so the official judge decides.</div>')
        test_runs = [r for q in load_questions("test") for r in runs.get(q["question_id"], [])]
        if test_runs:
            answered = sum(r["result"].get("status") == "answered" for r in test_runs)
            out.append(f'<div class="prose"><h3>Test run</h3><p>{answered} of {len(load_questions("test"))} test '
                       f'questions answered so far. Expected answers for test aren\'t released.</p></div>')
        for split_name, title in (("dev", "Finding the right conversations, 526 practice questions"),
                                  ("dev-large-130", "The same, with 10 times longer histories (130 questions)")):
            paths = {v: ROOT / "metrics" / f"retrieval-{split_name}-v{v}.json" for v in (1, 2)}
            if not all(p.exists() for p in paths.values()):
                continue
            data = {v: json.loads(p.read_text(encoding="utf-8"))["rows"] for v, p in paths.items()}
            rows = {TYPE_NAMES.get(k, k): {"n": data[1][k]["n"], **{f"{m} v{v}": data[v][k].get(m)
                                                                    for m in ("hit@5", "ans∈ev") for v in (1, 2)}}
                    for k in data[1]}
            out.append(f'<div class="prose"><h3>{title}</h3></div>')
            out.append(table(rows, [("n", "Questions"), ("hit@5 v1", "Found, v1"), ("hit@5 v2", "Found, v2"),
                                    ("ans∈ev v1", "Answer sent, v1"), ("ans∈ev v2", "Answer sent, v2")]))
            out.append('<div class="tcap">Found: a right conversation is in the top 5. Answer sent: the expected '
                       'answer is in the passages sent to Reactor. Measured offline (eval_retrieval.py).</div>')
        st.html("".join(out) or '<div class="sources-intro">No results saved yet.</div>')

# ---------- sidebar: technical details ----------

with details_slot:
    st.markdown("#### Details")
    rows = [("Question id", qid), ("History", f"{base['history_id']}, {n_sessions} conversations"),
            ("Retrieval", f"v{version}, {len(evidence)} passages, {sent_chars:,} characters "
                          f"(about {sent_chars / 4:,.0f} tokens; full history about {history_chars / 4:,.0f})")]
    if result:
        rows += [("Answer from", "a live Reactor call" if live_result else run["file"]),
                 ("Status", result.get("status", "-")),
                 ("Latency", f"{result['latency_s']:.0f} s" if result.get("latency_s") is not None else "-"),
                 ("Citations from", cite_source)]
        if g and prediction:
            f1 = score_question(prediction, g)["evidence_f1"]
            rows.append(("Evidence F1", "-" if f1 is None else f"{f1:.2f}"))
    else:
        rows.append(("Answer", f"none saved for v{version}"))
    st.html('<dl class="adv">' + "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in rows)
            + "</dl>")
