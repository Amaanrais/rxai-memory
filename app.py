"""Recall: ask about anything you've told an assistant, across months of chat history. A Streamlit demo.

    streamlit run app.py --server.address localhost          # saved Reactor answers only: never calls Reactor
    REACTOR_LIVE=1 streamlit run app.py --server.address localhost
        # adds a "Get a live answer" button (REACTOR_BASE_URL, REACTOR_AGENT_ID, REACTOR_ACCESS_KEY from the
        # environment; the key is never shown)

Two pages: Chat (a chat-first start screen with suggestions, then the conversation; each reply opens its Sources
and "How I answered" in a dialog) and About (for businesses, how it works, results). Answers come from the run
files in the repo root: every results-*.jsonl with its <name>.evidence.jsonl, as the drivers write them.
Retrieval runs live for any question. Advanced settings sit in the collapsed sidebar.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

from make_predictions import choose, predict
from retrieval import STOPWORDS, HistoryIndex, is_name_question, route, stem, tokenize
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
# below these, retrieval found nothing worth sending: the lowest top-1 score of any answerable question is 8.3 on dev
# and 14.6 on dev-large, and its smallest evidence 2,701 characters; gibberish scores 0, "hello" about 6
MIN_TOP_SCORE = 6.0
MIN_EVIDENCE_CHARS = 200
SET_NAMES = {"dev": "practice set", "test": "test set"}

# the suggestions on the start screen: the user's own words, the dev question they stand for, whether it runs on v2
CHIPS = [("What did I bake for my niece's party?", "q_4feb114c6944", False),
         ("Where's the Ethereal Dreams painting now?", "q_415adf25cebc", False),
         ("Where did I present my undergrad poster?", "q_43824099a143", True),  # the trap: v2 names missing words
         ("What 2FA methods did you mention?", "q_14252c64d73d", False)]
# a first name for each demo user (the histories are anonymous); other users get one from NAMES by their id
PEOPLE = {"h_e6d0b8417122": "Alex", "h_805dae031df3": "Sam", "h_c2c3efa33e8e": "Priya", "h_2bbadf4a2bcd": "Jordan"}
NAMES = ["Mateo", "Noor", "Lena", "Kofi", "Hana", "Ravi", "Zoe", "Elif", "Tomás", "Aiko", "Femi", "Ines"]

# the mark: a speech bubble with a highlighter stroke
MARK_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><circle cx="20" cy="20" r="20" fill="#0e5a5f"/>'
            '<path d="M11 13.5a3 3 0 0 1 3-3h12a3 3 0 0 1 3 3v8.5a3 3 0 0 1-3 3h-7.2l-4.6 3.8v-3.8H14a3 3 0 0 1-3-3z" '
            'fill="#fff"/><rect x="14.5" y="15.2" width="11" height="3.4" rx="1.2" fill="#ffe68a"/>'
            '<rect x="14.5" y="19.6" width="7" height="1.6" rx="0.8" fill="#9fb9ba"/></svg>')
# the browser tab's icon
MARK_URL = "data:image/svg+xml;base64," + __import__("base64").b64encode(MARK_SVG.encode()).decode()
# the wordmark: "Recall" in ink with a teal full stop
WORDMARK = 'Recall<span class="dot">.</span>'

st.set_page_config(page_title="Recall", page_icon=MARK_URL, layout="centered",
                   initial_sidebar_state="collapsed")

# ---------- style ----------
# The system font stack, 17px body, weights 400 and 600, one teal accent for every action, no shadows on cards or
# buttons, pills for actions and inputs, a frosted bar for what floats, and a soft teal gradient at the bottom of the
# start screen. The highlighter marks cited text in Sources only.
st.html("""
<style>
:root {
  --canvas: #f7f8f8; --surface: #ffffff; --ink: #1d1d1f; --muted: #5f6670; --faint: #6e747c;
  --hairline: rgba(0, 0, 0, 0.08); --accent: #0e5a5f; --accent-soft: #e2eeee; --accent-ring: #1b7a80;
  --mark: rgba(255, 230, 138, 0.6); --notice-bg: #f5f1e1; --notice-ink: #5c4a12;
  --font: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
}
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded),
[role="dialog"], [role="dialog"] *:not([data-testid="stIconMaterial"]) { font-family: var(--font); }
[role="dialog"] h2 { font-size: 21px; font-weight: 600; letter-spacing: -0.01em; }
.stApp { color: var(--ink); background:
  radial-gradient(60% 50% at 50% 110%, rgba(14, 90, 95, 0.17), transparent 70%),
  radial-gradient(42% 40% at 80% 104%, rgba(120, 196, 186, 0.22), transparent 70%),
  radial-gradient(38% 34% at 22% 104%, rgba(176, 222, 214, 0.22), transparent 70%),
  var(--canvas); background-attachment: fixed; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
.stDeployButton, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container { max-width: 760px; padding: 48px 24px 32px; }
[data-testid="stSidebar"] { background: var(--canvas); border-right: 1px solid var(--hairline); }
*:focus-visible { outline: 2px solid var(--accent-ring) !important; outline-offset: 2px; }
p, li { font-size: 17px; line-height: 1.47; }

/* the wordmark: small and fixed in the top-left corner on every screen, large over the start screen's title */
.st-key-brand { position: fixed; top: 17px; left: 22px; width: auto; z-index: 1000001; }
[data-testid="stToolbar"] > div:first-child { padding-left: 82px; }  /* the sidebar arrow and pages after it */
.wordmark { font-size: 20px; line-height: 1; font-weight: 700; letter-spacing: -0.02em; color: var(--accent); }
.wordmark .dot { color: var(--accent-ring); }
.hero .wordmark { font-size: 40px; letter-spacing: -0.03em; }
.about-head { margin: 0 0 8px; }
.about-head .wordmark { font-size: 40px; letter-spacing: -0.03em; }
.about-head .line { font-size: 21px; line-height: 1.3; font-weight: 600; letter-spacing: -0.01em; color: var(--muted);
  margin-top: 6px; }

/* the start screen */
.hero { display: flex; flex-direction: column; align-items: center; text-align: center; padding-top: 14vh; }
.hero h1 { font-size: 30px; line-height: 1.15; font-weight: 600; letter-spacing: -0.02em; color: var(--ink);
  margin: 14px 0 0; padding: 0; text-wrap: balance; }
.st-key-who { margin-top: -6px; }
.st-key-who button p, .st-key-whochat button p { font-size: 15px; color: var(--muted); }
.st-key-who button:hover p, .st-key-whochat button:hover p { color: var(--accent); }
.suggest { font-size: 14px; color: var(--muted); margin: 0 0 8px 2px; }

/* the bottom bar: suggestions over a pill input, on frosted glass so the conversation scrolls under it */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent; }
[data-testid="stBottomBlockContainer"] { max-width: 760px; padding-bottom: 28px; }
[data-testid="stBottom"] { background: rgba(247, 248, 248, 0.55); -webkit-backdrop-filter: saturate(180%) blur(20px);
  backdrop-filter: saturate(180%) blur(20px); }
.st-key-chips .stButton > button { background: rgba(255, 255, 255, 0.78); border: 1px solid var(--hairline);
  border-radius: 12px; min-height: 94px; padding: 12px 14px; box-shadow: none; color: var(--ink);
  justify-content: flex-start; align-items: flex-start; text-align: left; transition: transform 120ms ease; }
.st-key-chips .stButton > button > div, .st-key-chips .stButton > button > div > span { justify-content: flex-start;
  width: 100%; }
.st-key-chips .stButton > button p { font-size: 14px; line-height: 1.35; white-space: normal; text-align: left; }
.st-key-chips .stButton > button:hover { border-color: var(--accent); color: var(--ink); background: #fff; }
.stButton > button:active { transform: scale(0.97); }
[data-testid="stChatInput"] { border-radius: 9999px; border: 1px solid var(--hairline); background: var(--surface);
  box-shadow: none; padding: 4px 6px 4px 14px; }
[data-testid="stChatInput"] > div { border: none !important; border-radius: 9999px; background: transparent; }
[data-testid="stChatInputSubmitButton"] { border-radius: 9999px; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInput"] textarea { font-size: 17px; padding-left: 8px; }
[data-testid="stChatInputSubmitButton"] { color: var(--accent); }

/* the conversation */
[data-testid="stChatMessage"] { background: transparent; padding: 6px 0; gap: 12px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessageAvatarCustom"], [data-testid="stChatMessageAvatarAssistant"] {
  background: var(--accent-soft); color: var(--accent); border-radius: 9999px; }
.you { margin-left: auto; width: fit-content; max-width: 80%; background: var(--accent); color: #fff;
  border-radius: 20px 20px 6px 20px; padding: 10px 16px; font-size: 17px; line-height: 1.4; }
.reply { font-size: 17px; line-height: 1.47; color: var(--ink); }
.reply .answer { font-size: 22px; line-height: 1.3; font-weight: 600; letter-spacing: -0.01em; }
.reply .quiet { color: var(--muted); }
.reply .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.reply .chip { font-size: 13px; color: var(--accent); background: var(--accent-soft); border-radius: 9999px;
  padding: 4px 11px; white-space: nowrap; }
.typing { display: inline-flex; gap: 5px; padding: 10px 2px; }
.typing span { width: 7px; height: 7px; border-radius: 50%; background: var(--faint); opacity: 0.35;
  animation: blink 1s infinite ease-in-out; }
.typing span:nth-child(2) { animation-delay: 0.15s; } .typing span:nth-child(3) { animation-delay: 0.3s; }
@keyframes blink { 50% { opacity: 1; } }
.fresh { animation: rise 250ms ease-out both; }
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
[class*="st-key-act-"] .stButton > button { padding: 2px 0; min-height: 0; }
[class*="st-key-act-"] .stButton > button p { font-size: 15px; color: var(--accent); }
[class*="st-key-act-"] { gap: 18px; }
.whochat { font-size: 14px; color: var(--muted); text-align: center; }

/* dialogs */
[role="dialog"] { border-radius: 18px; animation: rise 220ms ease-out both; }
.sources-intro { font-size: 15px; color: var(--muted); margin: 0 0 10px; }
.hint { background: var(--notice-bg); color: var(--notice-ink); border-radius: 12px; padding: 12px 14px;
  font-size: 15px; line-height: 1.5; margin: 12px 0; }
details.src { border-top: 1px solid var(--hairline); padding: 14px 0; }
details.src:last-of-type { border-bottom: 1px solid var(--hairline); }
details.src summary { list-style: none; cursor: pointer; border-radius: 6px; }
details.src summary::-webkit-details-marker { display: none; }
.src .when { font-size: 15px; font-weight: 600; color: var(--ink); }
.src .when .ago { font-weight: 400; color: var(--muted); }
.src .when .tag { font-weight: 400; color: var(--accent); margin-left: 6px; }
.src .snippet { font-size: 15px; line-height: 1.5; color: var(--ink); margin-top: 4px; }
.src .snippet .open { color: var(--muted); font-size: 14px; margin-left: 6px; white-space: nowrap; }
details.src[open] .open { display: none; }
.src .turns { margin-top: 12px; padding: 14px 16px; background: #f5f5f7; border-radius: 12px; max-height: 380px;
  overflow-y: auto; }
.src .turn { margin-bottom: 12px; } .src .turn:last-child { margin-bottom: 0; }
.src .who { font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 2px; }
.src .said { font-size: 15px; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
.src .gap { color: var(--muted); font-size: 14px; margin-bottom: 12px; }
mark.hl { background: var(--mark); color: inherit; padding: 0 2px; border-radius: 2px;
  -webkit-box-decoration-break: clone; box-decoration-break: clone; }
.how { font-size: 16px; line-height: 1.5; }
.how h4 { font-size: 15px; font-weight: 600; margin: 18px 0 4px; padding: 0; }
.how h4:first-child { margin-top: 0; }
.how .big { font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
.how .muted { color: var(--muted); }
.bar { height: 4px; background: rgba(0, 0, 0, 0.06); border-radius: 2px; margin: 8px 0 2px; overflow: hidden; }
.bar span { display: block; height: 4px; background: var(--accent); border-radius: 2px; min-width: 3px; }

/* About */
.prose { font-size: 17px; line-height: 1.47; color: var(--ink); }
.prose h2 { font-size: 30px; font-weight: 600; letter-spacing: -0.02em; margin: 8px 0 6px; padding: 0; }
.prose h3 { font-size: 21px; font-weight: 600; letter-spacing: -0.01em; margin: 26px 0 6px; padding: 0; }
.prose ol, .prose ul { padding-left: 22px; margin: 0; } .prose li { margin-bottom: 8px; }
.prose .muted { color: var(--muted); }
.values { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }
.value { background: var(--surface); border: 1px solid var(--hairline); border-radius: 18px; padding: 20px 22px; }
.value .num { font-size: 34px; font-weight: 600; letter-spacing: -0.02em; color: var(--accent); line-height: 1.1; }
.value .what { font-size: 17px; font-weight: 600; margin-top: 6px; }
.value .how { font-size: 15px; color: var(--muted); margin-top: 4px; }
table.t { border-collapse: collapse; width: 100%; font-size: 15px; margin: 6px 0 4px; }
table.t, table.t th, table.t td { font-family: var(--font) !important; }
table.t th { font-weight: 400; color: var(--muted); text-align: right; padding: 6px 8px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.12); white-space: nowrap; }
table.t td { padding: 8px; text-align: right; border-bottom: 1px solid var(--hairline);
  font-variant-numeric: tabular-nums; }
table.t th:first-child, table.t td:first-child { text-align: left; padding-left: 0; width: 34%;
  font-variant-numeric: normal; }
table.t tr.all td { font-weight: 600; }
.tcap { font-size: 13px; color: var(--muted); margin-bottom: 18px; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--hairline); }
.stTabs [data-baseweb="tab"] { padding: 10px 0; } .stTabs [data-baseweb="tab"] p { font-size: 15px; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.adv dt { font-size: 13px; color: var(--muted); margin-top: 10px; } .adv { margin: 0; padding: 0; }
.adv dd { font-size: 14px; margin: 2px 0 0; overflow-wrap: anywhere; }

@media (max-width: 640px) {
  .block-container { padding: 24px 16px 16px; }
  .hero { padding-top: 6vh; } .hero h1 { font-size: 24px; } .values { grid-template-columns: 1fr; }
  .st-key-chips [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 8px; }
  .st-key-chips [data-testid="stColumn"] { flex: 1 1 calc(50% - 8px) !important; min-width: calc(50% - 8px) !important; }
  .st-key-chips .stButton > button { min-height: 72px; }
}
@media (prefers-reduced-motion: reduce) {
  .fresh, [role="dialog"], .typing span { animation: none !important; }
  .stButton > button:active { transform: none; }
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
    # v3 as submitted (run_v3_test.py): no missing-words hint, lenient abstain line
    text = (prompt(question) if version == 1 else prompt_v2(question, missing) if version == 2
            else prompt_v2(question, [], True))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evidence.json"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        extra = ["--history", histories[key], "--question", text, "--question-date", question_date,
                 "--question-id", qid, "--max-chars", str(ROUTED_MAX_CHARS if route(question, version)[1] > 12000 else MAX_CHARS)]
        if evidence:
            extra += ["--evidence", str(path)]
        return client(args, *extra)




# ---------- people and turns ----------

def person(history_id: str) -> str:
    """A first name for an anonymous history: the demo users have fixed ones, the rest one from NAMES by id."""
    if history_id in PEOPLE:
        return PEOPLE[history_id]
    return NAMES[int(hashlib.sha1(history_id.encode()).hexdigest(), 16) % len(NAMES)]


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


st.sidebar.markdown("### Advanced")
split = st.sidebar.radio("Dataset", list(SPLITS), horizontal=True,
                         help="dev is the practice set, with expected answers; test is the scored set")
version = int(st.sidebar.radio("Retrieval version", ["v1", "v2", "v3"], index=2, horizontal=True,
                                help="v3 produced the submitted answers: v2 plus name search and a wider budget "
                                     "for name questions. v2 adds focused trimming, more room for a leading "
                                     "conversation, time sentences and the missing-words note. v1 was the first "
                                     "test run. The trap suggestion always uses v2.")[1])
live_ready = LIVE and all(os.environ.get(k) for k in LIVE_ENV)
st.sidebar.markdown("#### Live answers")
if live_ready:
    st.sidebar.write("On. Asking starts a new Reactor history on this agent, which interrupts a benchmark running "
                     "on it.")
elif LIVE:
    st.sidebar.write("Not configured: set " + ", ".join(k for k in LIVE_ENV if not os.environ.get(k)) + ".")
else:
    st.sidebar.write("Off. Replies come from our saved runs. Start the app with REACTOR_LIVE=1 to ask Reactor.")

questions = load_questions(split)
by_id = {q["question_id"]: q for q in questions}
runs = load_runs()
state = st.session_state
if state.get("split") != split:  # a new dataset starts a new chat, as the first demo user
    state.split, state.turns, state.animate = split, [], None
    first = next((qid for _, qid, _ in CHIPS if qid in by_id), None)
    state.history = by_id[first]["history_id"] if first else questions[0]["history_id"]


def user_questions(history_id: str) -> list[dict]:
    return [q for q in questions if q["history_id"] == history_id]


def add_turn(text: str, qid: str | None = None, v2: bool = False) -> None:
    """A question into the chat. A suggestion stands for its dataset question (and switches to that user); typed
    text is matched to one of this user's dataset questions, or else asked as is, on this user's latest date."""
    if qid is None:
        match = next((q for q in user_questions(state.history)
                      if q["question"].strip().lower() == text.strip().lower()), None)
        qid = match["question_id"] if match else None
    if qid:
        q = by_id[qid]
        state.history, question, date = q["history_id"], q["question"], q["question_date"]
    else:
        question = text
        date = max((q["question_date"] for q in user_questions(state.history)), key=date_of)
    state.turns.append({"text": text, "question": question, "qid": qid, "history": state.history, "date": date,
                        "version": 2 if v2 else version, "split": split, "trap": v2})
    state.animate = len(state.turns) - 1


def resolve(turn: dict) -> SimpleNamespace:
    """Everything a reply needs: the saved (or live) answer, the evidence sent with it, and how much was read."""
    index, history_chars, n_sessions = load_index(turn["split"], turn["history"])
    qid, v = turn["qid"], turn["version"]
    # a version 3 run counts for version 2 on any other than a name question: its evidence is version 2's
    same = {v, 3} if v == 2 and not is_name_question(turn["question"]) else {v}
    run = next((r for r in reversed(runs.get(qid, [])) if r["version"] in same), None) if qid else None
    live_key = (turn["split"], turn["history"], turn["question"], turn["date"], v)
    live_result = state.get("live", {}).get(live_key)
    if run and run["logged"]:
        evidence = run["logged"]["evidence"]
        missing = run["logged"].get("missing") or (index.missing_words(turn["question"], evidence) if v >= 2 else [])
    else:
        evidence = index.evidence(turn["question"], question_date=turn["date"], version=v)
        missing = index.missing_words(turn["question"], evidence) if v >= 2 else []
    result = live_result or (run["result"] if run else None)
    prediction, cite_source = predict(qid or "q_custom", result, {"evidence": evidence}) if result else (None, "")
    evidence_chars = sum(len(e["text"]) for e in evidence)
    top = index.search(turn["question"], k_sessions=1)
    no_evidence = (not evidence or (top[0][1] if top else 0.0) < MIN_TOP_SCORE
                   or evidence_chars < MIN_EVIDENCE_CHARS)
    return SimpleNamespace(index=index, history_chars=history_chars, n_sessions=n_sessions, run=run,
                           live_key=live_key, live=bool(live_result), evidence=evidence, missing=missing,
                           result=result, prediction=prediction, cite_source=cite_source,
                           cited=set(prediction["citations"]) if prediction else set(),
                           sent_chars=evidence_chars + len(turn["question"]), no_evidence=no_evidence,
                           gold=load_gold(turn["split"]).get(qid) if qid else None)


def reply_html(r: SimpleNamespace) -> str:
    """The reply in plain words: the answer and up to two sources, "I don't have that", or what Recall would do."""
    p = r.prediction
    if p is None and r.no_evidence:
        return "<div>I don't have that in our past conversations.</div>"
    if p is None:
        n = len(r.evidence)
        rest = " Get a live answer below." if live_ready else " Live answers are off right now."
        return f'<div class="quiet">I\'d look in {plural(n, "past conversation")} for this.{rest}</div>'
    if r.result.get("status") != "answered":
        return "<div>I couldn't answer that in time, so I won't guess.</div>"
    if p["abstain"]:
        return "<div>I don't have that in our past conversations.</div>"
    dates = {e["id"]: e["date"] for e in r.evidence}
    sources = sorted((s for s in r.cited if s in dates), key=lambda s: date_of(dates[s]))
    chips = "".join(f'<span class="chip">From our chat on {day(dates[s], year=False)}</span>' for s in sources[:2])
    if len(sources) > 2:
        chips += f'<span class="chip">and {plural(len(sources) - 2, "more")}</span>'.replace("mores", "more")
    return f'<div class="answer">{html.escape(p["answer"])}</div>' + (f'<div class="chips">{chips}</div>' if chips
                                                                       else "")


def sources_html(turn: dict, r: SimpleNamespace) -> str:
    """The conversations sent, oldest first: collapsed rows with a snippet, cited ones under the highlighter."""
    words = keywords(turn["question"])
    shown = [] if r.prediction is None and r.no_evidence else r.evidence
    if not shown:
        return '<div class="sources-intro">No past conversation matches this closely enough to send.</div>'
    verb = "would send" if r.prediction is None else "sent"
    parts = [f'<div class="sources-intro">{plural(len(shown), "conversation")} Recall {verb} to Reactor, oldest '
             f'first.{" Highlighted: the ones the answer comes from." if r.cited else ""}</div>']
    for e in shown:
        days, recent, body = parse_excerpt(e["text"])
        is_cited = e["id"] in r.cited
        turns = turns_of(body)
        snippet = best_sentence(" ".join(t for who, t in turns if who), words)
        ago = ("the day of the question" if days == 0 else f"{plural(days, 'day')} before the question"
               ) if days is not None else ""
        tags = (' <span class="tag">cited</span>' if is_cited else "") + \
               (' <span class="tag">most recent</span>' if recent else "")
        line = clip_line(snippet)
        line = f'<mark class="hl">{html.escape(line)}</mark>' if is_cited else html.escape(line)
        inner = ""
        for who, text in turns:
            if not who:
                inner += '<div class="gap">…</div>'
                continue
            said = highlighted(text, snippet) if is_cited else html.escape(text)
            inner += f'<div class="turn"><div class="who">{html.escape(who)}:</div><div class="said">{said}</div></div>'
        parts.append(f'<details class="src"><summary><div class="when">{day(e["date"])}'
                     f'{", " + ago if ago else ""}{tags}</div><div class="snippet">{line}'
                     f'<span class="open">Show conversation</span></div></summary>'
                     f'<div class="turns">{inner}</div></details>')
    return "".join(parts)


CITE_WORDS = {
    "answer labels": "Reactor named the passages it used on its answer line, so those conversations are the sources.",
    "reasoning labels": "Reactor named the passages it used in its reasoning, so those conversations are the sources.",
    "support (exact)": "Reactor named no passages, so Recall cited only the passages that contain its answer.",
    "support (partial)": "Reactor named no passages, so Recall cited only the passages holding most of its answer's "
                         "words.",
    "support (unspecific)": "Reactor named no passages, and its answer appears in too many to single any out, so "
                            "Recall cites none.",
    "support (unsupported)": "Reactor named no passages and none contains its answer, so Recall cites none.",
    "abstain": "No sources: Recall cites nothing when it says it doesn't know.",
}


def how_html(turn: dict, r: SimpleNamespace) -> str:
    who = person(turn["history"])
    out = ['<div class="how">']
    out.append("<h4>What I read</h4>")
    if r.no_evidence and r.prediction is None:
        out.append(f"<div>{who} has {plural(r.n_sessions, 'past conversation')} in the {SET_NAMES[turn['split']]}. "
                   f"Nothing in them matched closely enough, so nothing would be sent.</div>")
    else:
        share = r.sent_chars / r.history_chars
        out.append(f'<div class="big">{short(r.sent_chars)} of {short(r.history_chars)} characters</div>'
                   f'<div class="bar" role="img" aria-label="{share:.1%} of the history"><span style="width:'
                   f'{min(share, 1):.2%}"></span></div>'
                   f'<div class="muted">{r.history_chars / r.sent_chars:.0f}× less than {who}\'s whole history: '
                   f'{plural(len(r.evidence), "passage")} from {plural(r.n_sessions, "past conversation")}, about '
                   f'{r.sent_chars / 4:,.0f} tokens instead of {r.history_chars / 4:,.0f}.</div>')
    if turn["qid"] and turn["text"].strip() != turn["question"].strip():
        out.append(f"<h4>The question</h4><div>You asked “{html.escape(turn['text'])}”. I looked it up as "
                   f"“{html.escape(turn['question'])}”, asked on {day(turn['date'])}.</div>")
    out.append("<h4>Checking the sources</h4>")
    if r.prediction is None:
        out.append('<div class="muted">Nothing to check yet: this question hasn\'t been sent to Reactor.</div>')
    else:
        out.append(f"<div>{CITE_WORDS.get(r.cite_source, html.escape(r.cite_source))}</div>")
    if r.missing:
        out.append(f'<div class="hint">Words from the question that appear in none of these conversations: '
                   f'<b>{html.escape(", ".join(r.missing))}</b>. Reactor is told to say it doesn\'t know if the '
                   f'answer depends on them. This is how near-miss traps are caught.</div>')
    if r.gold and r.prediction is not None:
        verdict = score_question(r.prediction, r.gold)["answer"]
        expected = "It isn't in the history." if r.gold["abstain"] else r.gold["answer"]
        line = {"correct": "matches it.", "unjudged": "isn't an exact match, so the official judge decides.",
                "wrong": "doesn't match it."}[verdict]
        out.append(f"<h4>Expected answer</h4><div>{html.escape(expected)} <span class=\"muted\">The reply "
                   f"{line}</span></div>")
    meta = [f"retrieval v{turn['version']}" + (" (used automatically for this trap question)" if turn["trap"]
                                                else "")]
    if r.result and r.result.get("latency_s") is not None:
        meta.append(f"Reactor took {r.result['latency_s']:.0f} s")
    meta.append("a live answer" if r.live else ("saved from our run" if r.run else "not sent"))
    out.append(f'<h4>Details</h4><div class="muted">{html.escape(", ".join(meta)).capitalize()}.</div></div>')
    return "".join(out)


@st.dialog("Sources", width="large")
def sources_dialog(i: int) -> None:
    turn = state.turns[i]
    st.html(sources_html(turn, resolve(turn)))


@st.dialog("How I answered", width="large")
def how_dialog(i: int) -> None:
    turn = state.turns[i]
    st.html(how_html(turn, resolve(turn)))


@st.dialog("Switch user")
def switch_dialog() -> None:
    """Choose whose history to chat with: the demo users first, then everyone with a saved answer, then the rest."""
    order = list(dict.fromkeys([h for h in PEOPLE if any(q["history_id"] == h for q in questions)]
                               + [q["history_id"] for q in questions if q["question_id"] in runs]
                               + [q["history_id"] for q in questions]))
    first_q = {}
    for q in questions:
        first_q.setdefault(q["history_id"], q["question"])
    choice = st.selectbox("Who are you chatting as?", order, index=order.index(state.history),
                          format_func=lambda h: f"{person(h)}, who once asked “{first_q[h][:70]}”")
    if st.button("Switch", type="primary"):
        state.history, state.turns, state.animate = choice, [], None
        st.rerun()


def ask_chip(text: str, qid: str, v2: bool) -> None:
    add_turn(text, qid, v2)


def new_chat() -> None:
    state.turns, state.animate = [], None


# ---------- pages ----------

def chat_page() -> None:
    bottom = getattr(st, "bottom", None) or getattr(st, "_bottom", None)  # the chat input's bar: chips go above it
    with bottom if bottom is not None else st.container():
        chips_slot = st.empty()
    prompt = st.chat_input("Ask anything you've told me")
    if prompt and prompt.strip():
        add_turn(prompt.strip())

    _, _, n_sessions = load_index(split, state.history)
    who_line = f"Chatting as {person(state.history)} · {plural(n_sessions, 'past conversation')}"

    if not state.turns:
        chips = [c for c in CHIPS if c[1] in by_id] or [
            (q["question"], q["question_id"], False) for q in user_questions(state.history)[:4]]
        with chips_slot.container():
            st.html('<div class="suggest">Try asking</div>')
            with st.container(key="chips"):
                for col, (text, qid, v2) in zip(st.columns(len(chips)), chips):
                    col.button(text, key=f"chip-{qid}", width="stretch", on_click=ask_chip, args=(text, qid, v2))
        st.html(f'<div class="hero"><div class="wordmark">{WORDMARK}</div>'
                '<h1>Ask about anything you\'ve told me.</h1></div>')
        with st.container(key="who", horizontal=True, horizontal_alignment="center"):
            if st.button(who_line, type="tertiary", help="Switch to another user's chat history"):
                switch_dialog()
        return

    with st.container(key="whochat", horizontal=True, horizontal_alignment="center"):
        if st.button(who_line, type="tertiary", key="who-chat", help="Switch to another user's chat history"):
            switch_dialog()
        st.button("New chat", type="tertiary", key="new-chat", on_click=new_chat)
    for i, turn in enumerate(state.turns):
        with st.chat_message("user"):
            st.html(f'<div class="you">{html.escape(turn["text"])}</div>')
        with st.chat_message("assistant", avatar=":material/history:"):
            fresh = state.get("animate") == i
            if fresh:  # a short typing indicator, then the reply rises in
                typing = st.empty()
                typing.html('<div class="typing" aria-label="Recall is typing"><span></span><span></span>'
                            '<span></span></div>')
                time.sleep(0.6)
                typing.empty()
                state.animate = None
            r = resolve(turn)
            st.html(f'<div class="reply{" fresh" if fresh else ""}">{reply_html(r)}</div>')
            with st.container(key=f"act-{i}", horizontal=True):
                if st.button("Sources", key=f"src-{i}", type="tertiary"):
                    sources_dialog(i)
                if st.button("How I answered", key=f"how-{i}", type="tertiary"):
                    how_dialog(i)
                if live_ready and r.prediction is None and not r.no_evidence and st.button(
                        "Get a live answer", key=f"live-{i}", type="tertiary"):
                    with st.spinner("Reactor is answering, usually within a minute..."):
                        answer = ask_reactor(turn["split"], turn["history"], turn["qid"] or "q_custom",
                                             turn["question"], turn["date"], turn["version"], r.evidence, r.missing)
                    state.setdefault("live", {})[r.live_key] = answer
                    st.rerun()


def sources_cited_share() -> tuple[int, int]:
    """(answers that name their sources, answers), over every saved run."""
    named = total = 0
    for qid, rs in runs.items():
        for r in rs:
            prediction, _ = predict(qid, r["result"], r["logged"])
            if not prediction["abstain"]:
                total += 1
                named += bool(prediction["citations"])
    return named, total


def about_page() -> None:
    st.html(f'<div class="about-head"><div class="wordmark">{WORDMARK}</div>'
            '<div class="line">A memory layer for AI assistants.</div></div>')
    business, how, results = st.tabs(["For businesses", "How it works", "Results"])
    with business:
        named, total = sources_cited_share()
        st.html(f"""<div class="prose">
<h2>A memory your customers can check.</h2>
<p>Recall gives an AI assistant or a support team a reliable memory of months of past conversations: one answer, the
conversations it came from, and an honest “I don't know” when the history doesn't say.</p>
<h3>Who it's for</h3>
<ul>
<li><b>Support teams</b> picking up a customer they've talked to for months: what was promised, what changed, what
was already tried.</li>
<li><b>AI assistants with long-term memory</b> that must recall what a user said weeks ago without re-reading
everything.</li>
<li><b>Anyone who has to show their work</b>: every answer points back to the conversation it came from.</li>
</ul>
<div class="values">
<div class="value"><div class="num">~200×</div><div class="what">Fewer characters per question</div>
<div class="how">A median of 8.2k characters sent per question, against about 1.66 million in a full user history
(the first 37 test questions).</div></div>
<div class="value"><div class="num">~20×</div><div class="what">Less than loading a history once</div>
<div class="how">All 9 or 10 of a user's questions together still send 19 to 23 times less than loading their whole
history once (test histories).</div></div>
<div class="value"><div class="num">{named} of {total}</div><div class="what">Answers with their sources</div>
<div class="how">Every answer names the conversations it came from. Without Reactor's own labels, Recall cites only
passages that contain the answer.</div></div>
<div class="value"><div class="num">50%</div><div class="what">Of trap questions flagged</div>
<div class="how">On unanswerable practice questions, v2 names the question's words that never appear, so Reactor
can say “I don't know” instead of guessing. It flags 12.5% of answerable ones.</div></div>
</div>
<p class="muted" style="margin-top:18px">Measured on the challenge's practice and test sets. Final answers always
come from Reactor.</p></div>""")
    with how:
        st.html("""<div class="prose">
<h3>What Recall does</h3>
<ol>
<li><b>Reads the whole history once.</b> Every message, yours and the assistant's, is indexed so it can be searched
in a fraction of a second.</li>
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
<li><b>Something the assistant said:</b> the assistant's messages are searched too.</li>
<li><b>Questions with no answer:</b> “I don't know” is an answer, and it comes with no sources.</li>
</ul></div>""")
    with results:
        dev_gold = load_gold("dev")
        out = []
        for name in sorted({r["file"] for rs in runs.values() for r in rs}):
            entries = [(q, r) for q, rs in runs.items() for r in rs if r["file"] == name and q in dev_gold]
            if not entries:
                continue
            scored = [score_question(predict(q, r["result"], r["logged"])[0], dev_gold[q]) for q, r in entries]
            rows = {TYPE_NAMES.get(k, k): v for k, v in summarize(scored).items()}
            out.append(f'<div class="prose"><h3>Answers on {len(entries)} practice questions</h3></div>')
            out.append(table(rows, [("n", "Questions"), ("correct", "Match"), ("unjudged", "For the judge"),
                                    ("abstention_recall", "Rightly said “don't know”"),
                                    ("evidence_f1", "Sources right")]))
            out.append('<div class="tcap">Scored with the organisers\' checker on one saved run. "For the judge": '
                       'not an exact match, so the official judge decides.</div>')
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
                       'answer is in the passages sent to Reactor. Measured offline.</div>')
        st.html("".join(out) or '<div class="sources-intro">No results saved yet.</div>')


with st.container(key="brand"):
    st.html(f'<div class="wordmark">{WORDMARK}</div>')
st.navigation([st.Page(chat_page, title="Chat", default=True),
               st.Page(about_page, title="About", url_path="about")], position="top").run()
