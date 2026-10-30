"""Turn-level BM25 retrieval over one dataset history, standard library only.

    from retrieval import HistoryIndex, retrieve
    index = HistoryIndex(history)            # one dict from histories.jsonl
    index.search(question_text)              # [(session_id, score)], best first
    index.evidence(question_text)            # excerpts in client.py's evidence format

Every turn of both roles is a chunk, prefixed ``user: `` / ``assistant: ``; turns over ``WINDOW`` characters are
split into windows of about that size. ``retrieve()`` is a drop-in for ``examples/run_benchmark.py``'s.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import date, datetime

WINDOW = 1500
NEIGHBOUR_CHARS = 600
LEAD_RATIO = 0.6  # version 2: a top session leads when the next scores under this share of it
TIME_CUE_LABEL = "user (elsewhere in this chat): "
RELATIVE_TIME = re.compile(
    r"\b(?:ago|yesterday|the other day|the day before"
    r"|(?:last|this past|earlier this|later this)\s+(?:week|weekend|month|year|night|summer|spring|fall|autumn|winter"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
K1, B = 1.2, 0.75

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being below between
both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his
how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves
s t d ll m re ve just also will shall may might must get got really yes ok okay
tell told telling remind reminded say said saying mention mentioned mentioning remember remembered recall
recalled discuss discussed previous previously earlier conversation chat
""".split())
TOKEN = re.compile(r"[a-z0-9]+")
# how a question refers back to the past, not what it asks about: never a missing word
FRAMING = frozenset("""
remind remember recall mention previous previously earlier before conversation chat talk discuss ask asked look
looking back wonder wondering curious help again thinking think going last time
""".split()) | frozenset("""conversation chatted talked discussed mentioned reminded""".split())


def stem(word: str) -> str:
    """Strip one of -ies, -ing, -ed, -es, -s, keeping a stem of at least three letters."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    out = []
    for word in TOKEN.findall(text.lower()):
        if word in STOPWORDS:
            continue
        word = stem(word)
        if word not in STOPWORDS:
            out.append(word)
    return out


class HistoryIndex:
    """BM25 over the turn chunks of one dataset history."""

    def __init__(self, history: dict):
        self.history = history
        self.sessions = history["sessions"]
        # chunks[i] = (session index, turn index, window index); chunk_text[i] its text with the role prefix
        self.chunks: list[tuple[int, int, int]] = []
        self.chunk_text: list[str] = []
        for si, session in enumerate(self.sessions):
            for ti, turn in enumerate(session["turns"]):
                prefix = f"{turn['role']}: "
                content = turn["content"] or ""
                for wi, start in enumerate(range(0, max(len(content), 1), WINDOW)):
                    self.chunks.append((si, ti, wi))
                    self.chunk_text.append(prefix + content[start:start + WINDOW])

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for ci, text in enumerate(self.chunk_text):
            counts: dict[str, int] = defaultdict(int)
            tokens = tokenize(text)
            for token in tokens:
                counts[token] += 1
            for token, tf in counts.items():
                self.postings[token].append((ci, tf))
            lengths.append(len(tokens))
        self.lengths = lengths
        self.avgdl = sum(lengths) / len(lengths) if lengths else 1.0
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        idfs = sorted(self.idf.values())
        self.median_idf = idfs[len(idfs) // 2] if idfs else 0.0
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)  # a word the history never uses

    def rank_chunks(self, question: str) -> list[tuple[int, float]]:
        """[(chunk index, BM25 score)] for every chunk matching a question term, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(question)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for ci, tf in self.postings[term]:
                norm = K1 * (1 - B + B * self.lengths[ci] / self.avgdl)
                scores[ci] += idf * tf * (K1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda s: -s[1])

    def _session_scores(self, ranked: list[tuple[int, float]]) -> list[tuple[int, float]]:
        per_session: dict[int, list[float]] = defaultdict(list)
        for ci, score in ranked[:100]:
            per_session[self.chunks[ci][0]].append(score)  # already best first
        scored = [(si, s[0] + 0.25 * sum(s[1:3])) for si, s in per_session.items()]
        return sorted(scored, key=lambda s: -s[1])

    def search(self, question: str, k_sessions: int = 5) -> list[tuple[str, float]]:
        """The ``k_sessions`` best sessions: best chunk + 0.25 x the next two chunks of that session."""
        scored = self._session_scores(self.rank_chunks(question))[:k_sessions]
        return [(self.sessions[si]["session_id"], score) for si, score in scored]

    def suggest_citations(self, question: str, ratio: float = 0.6, multi_k: int = 2, max_k: int = 5) -> list[str]:
        """Session ids to cite: the top one, plus each next one scoring >= ``ratio`` x the top.

        An aggregate question (``is_aggregate()``) cites at least ``multi_k``."""
        ranked = self.search(question, k_sessions=max_k)
        if not ranked:
            return []
        n = 1
        while n < len(ranked) and ranked[n][1] >= ratio * ranked[0][1]:
            n += 1
        if is_aggregate(question):
            n = max(n, min(multi_k, len(ranked)))
        return [sid for sid, _ in ranked[:n]]

    def evidence(self, question: str, budget: int | None = None, k_sessions: int | None = None,
                 per_session_hits: int = 2, question_date: str | None = None, version: int = 1,
                 lead: bool | None = None, time_cues: bool | None = None) -> list[dict]:
        """Excerpts of the top sessions, chronological: each session's best turns with one turn either side.

        ``budget`` and ``k_sessions`` default to the question's route (``route()``). With ``question_date``,
        each excerpt starts with how many days before it its session took place. The latest excerpt is
        marked ``[most recent]``: for a value that changed, the latest mention wins.

        ``version=2``: a hit turn too long for its room keeps the stretch with the most question words (``focus()``),
        not its beginning. (Dropping a session's second hit turn when it scored under 0.8x its first lowered the
        share of gold answers in the evidence on dev, so version 2 doesn't.) Also, unless turned off:
        - ``lead``: when the top session clearly leads (the next scores under ``LEAD_RATIO`` x it, as in
          ``suggest_citations()``), it gets up to half the budget, and the others share the rest;
        - ``time_cues``: for an aggregate question (``is_aggregate()``), each excerpt adds its session's user
          sentences that say when something happened ("last month", "three weeks ago"), wherever they are."""
        lead = version >= 2 if lead is None else lead
        time_cues = version >= 2 if time_cues is None else time_cues
        route_k, route_budget = route(question)
        budget = route_budget if budget is None else budget
        k_sessions = route_k if k_sessions is None else k_sessions
        ranked = self.rank_chunks(question)
        top = self._session_scores(ranked)[:k_sessions]
        if not top:
            return []
        share = budget // len(top)
        # best windows per (session, turn), in rank order
        hits: dict[int, dict[int, list[int]]] = defaultdict(dict)
        for ci, _ in ranked:
            si, ti, _ = self.chunks[ci]
            turns = hits[si]
            if ti in turns:
                turns[ti].append(ci)
            elif len(turns) < per_session_hits:
                turns[ti] = [ci]
        weights = {t: self.idf[t] for t in set(tokenize(question)) if t in self.idf} if version >= 2 else None
        cues = time_cues and is_aggregate(question)
        chronological = sorted(top)  # session order is chronological
        latest = chronological[-1][0]

        def excerpt(si: int, room: int) -> str:
            session = self.sessions[si]
            notes = []
            if question_date:
                days = days_before(session["date"], question_date)
                notes.append(f"({days} day{'' if days == 1 else 's'} before the question date)")
            if si == latest:
                notes.append("[most recent]")
            header = "\n".join(notes)
            said = self._time_sentences(si, room // 2) if cues else []
            extra = sum(len(s) + len(TIME_CUE_LABEL) + 1 for s in said)
            text = self._excerpt(si, hits[si], room - len(header) - 1 - extra, weights)
            said = [s for s in said if s not in text]
            text = "\n".join([text] + [TIME_CUE_LABEL + s for s in said])
            return f"{header}\n{text}" if header else text

        texts: dict[int, str] = {}
        if lead and len(top) > 1 and top[1][1] < LEAD_RATIO * top[0][1]:
            texts[top[0][0]] = excerpt(top[0][0], max(share, budget // 2))
            share = (budget - len(texts[top[0][0]])) // (len(top) - 1)
        for si, _ in chronological:
            if si not in texts:
                texts[si] = excerpt(si, share)
        return [{"id": self.sessions[si]["session_id"], "date": self.sessions[si]["date"], "text": texts[si]}
                for si, _ in chronological]

    def _time_sentences(self, si: int, room: int) -> list[str]:
        """The user's sentences in session ``si`` that say when something happened, in order, within ``room``."""
        found: list[str] = []
        used = 0
        for turn in self.sessions[si]["turns"]:
            if turn["role"] != "user":
                continue
            for sentence in SENTENCE.split(turn["content"]):
                sentence = clip(sentence.strip(), 300)
                if sentence and RELATIVE_TIME.search(sentence) and sentence not in found:
                    used += len(sentence) + len(TIME_CUE_LABEL) + 1
                    if used > room:
                        return found
                    found.append(sentence)
        return found

    def missing_words(self, question: str, evidence: list[dict], limit: int = 4) -> list[str]:
        """The question's content words found nowhere in ``evidence``, rarest first: those rarer than the
        history's median word, or absent from the history. Framing words (remind, previous, chat...) don't count."""
        sent = {t for e in evidence for t in tokenize(e["text"])}
        words = {}
        for word in TOKEN.findall(question.lower()):
            if word in STOPWORDS or word in FRAMING:
                continue
            term = stem(word)
            if term in STOPWORDS or term in FRAMING or term in sent or term in words:
                continue
            idf = self.idf.get(term, self.max_idf)
            if idf > self.median_idf:
                words[term] = (idf, word)
        return [word for _, word in sorted(words.values(), key=lambda w: -w[0])[:limit]]

    def _excerpt(self, si: int, hit_turns: dict[int, list[int]], share: int,
                 weights: dict[str, float] | None = None) -> str:
        turns = self.sessions[si]["turns"]
        if weights is None:
            hit_text = {ti: self._hit_text(turns[ti], cis) for ti, cis in hit_turns.items()}
        else:
            hit_text = {ti: f"{turns[ti]['role']}: {turns[ti]['content']}" for ti in hit_turns}
        neighbours = sorted({n for ti in hit_turns for n in (ti - 1, ti + 1)
                             if 0 <= n < len(turns) and n not in hit_turns})
        # the hits keep their text first; the neighbours share what is left, up to NEIGHBOUR_CHARS each
        hit_chars = sum(len(t) + 1 for t in hit_text.values())
        if hit_chars > share:
            cap = share // max(len(hit_text), 1) - 1
            if weights is None:
                hit_text = {ti: clip(t, cap) for ti, t in hit_text.items()}
            else:
                hit_text = {ti: focus(t, cap, weights) for ti, t in hit_text.items()}
            hit_chars = sum(len(t) + 1 for t in hit_text.values())
        room = share - hit_chars - 6 * len(neighbours)
        cap = min(NEIGHBOUR_CHARS, room // len(neighbours)) if neighbours else 0
        parts = dict(hit_text)
        if cap >= 80:
            for n in neighbours:
                parts[n] = clip(f"{turns[n]['role']}: {turns[n]['content']}", cap)
        lines, last = [], None
        for ti in sorted(parts):
            if last is not None and ti != last + 1:
                lines.append("...")
            lines.append(parts[ti])
            last = ti
        return clip("\n".join(lines), share)

    def _hit_text(self, turn: dict, chunk_ids: list[int]) -> str:
        """A hit turn whole, or for a split turn its matching windows in order."""
        if len(turn["content"]) <= WINDOW:
            return f"{turn['role']}: {turn['content']}"
        windows = sorted(self.chunks[ci][2] for ci in chunk_ids[:2])
        body = " ... ".join(turn["content"][w * WINDOW:(w + 1) * WINDOW] for w in windows)
        return f"{turn['role']}: {body}"


AGGREGATE = re.compile(r"\b(how many|total|all|both|combined|before|after|first|last|between|how long|ago)\b"
                       r"|\bwhich\b.*\bfirst\b")


def is_aggregate(question: str) -> bool:
    """A question that counts, totals, compares or orders: its answer usually spans several sessions."""
    return bool(AGGREGATE.search(question.lower()))


def route(question: str) -> tuple[int, int]:
    """(k_sessions, budget): 8 sessions in 24,000 characters for aggregate questions, else 5 in 12,000."""
    return (8, 24000) if is_aggregate(question) else (5, 12000)


def parse_date(value: str) -> date:
    """The calendar day of a dataset date, "2023/05/20 (Sat) 02:21"."""
    return datetime.strptime(value[:10], "%Y/%m/%d").date()


def days_before(session_date: str, question_date: str) -> int:
    return (parse_date(question_date) - parse_date(session_date)).days


def focus(text: str, limit: int, weights: dict[str, float]) -> str:
    """``text`` cut to ``limit`` characters around the stretch covering the most question words (each distinct word
    counted once, by idf), its role prefix kept, cuts marked "..."; without any question word, its beginning."""
    if len(text) <= limit:
        return text
    prefix, _, body = text.partition(": ")
    prefix += ": "
    room = limit - len(prefix) - 6  # two "..." marks
    hits = []
    for m in TOKEN.finditer(body.lower()):
        if m.group() not in STOPWORDS and stem(m.group()) in weights:
            hits.append((m.start(), stem(m.group())))
    if room <= 0 or not hits:
        return clip(text, limit)
    # slide over the hits: the window [hits[left], hits[right]] spanning under ``room`` with the best coverage
    counts: dict[str, int] = defaultdict(int)
    covered, best, best_span, left = 0.0, -1.0, (0, 0), 0
    for right, (pos, term) in enumerate(hits):
        counts[term] += 1
        if counts[term] == 1:
            covered += weights[term]
        while pos - hits[left][0] >= room:
            gone = hits[left][1]
            counts[gone] -= 1
            if not counts[gone]:
                covered -= weights[gone]
            left += 1
        if covered > best:
            best, best_span = covered, (hits[left][0], pos)
    # centre the slack around the winning stretch
    start = max(0, min(best_span[0] - (room - (best_span[1] - best_span[0])) // 2, len(body) - room))
    end = start + room
    return prefix + ("..." if start else "") + body[start:end] + ("..." if end < len(body) else "")


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."


def retrieve(question: dict, history: dict, k: int | None = None) -> list[dict]:
    """Drop-in for run_benchmark.py's ``retrieve()``: evidence from the ``k`` best sessions (default: routed)."""
    return HistoryIndex(history).evidence(question["question"], k_sessions=k,
                                          question_date=question.get("question_date"))
