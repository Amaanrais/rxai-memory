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

WINDOW = 1500
NEIGHBOUR_CHARS = 600
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

    def evidence(self, question: str, budget: int = 12000, k_sessions: int = 5,
                 per_session_hits: int = 2) -> list[dict]:
        """Excerpts of the top sessions, chronological: each session's best turns with one turn either side."""
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
        excerpts = []
        for si, _ in sorted(top):  # session order is chronological
            excerpts.append({"id": self.sessions[si]["session_id"], "date": self.sessions[si]["date"],
                             "text": self._excerpt(si, hits[si], share)})
        return excerpts

    def _excerpt(self, si: int, hit_turns: dict[int, list[int]], share: int) -> str:
        turns = self.sessions[si]["turns"]
        hit_text = {ti: self._hit_text(turns[ti], cis) for ti, cis in hit_turns.items()}
        neighbours = sorted({n for ti in hit_turns for n in (ti - 1, ti + 1)
                             if 0 <= n < len(turns) and n not in hit_turns})
        # the hits keep their text first; the neighbours share what is left, up to NEIGHBOUR_CHARS each
        hit_chars = sum(len(t) + 1 for t in hit_text.values())
        if hit_chars > share:
            cap = share // max(len(hit_text), 1) - 1
            hit_text = {ti: clip(t, cap) for ti, t in hit_text.items()}
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


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."


def retrieve(question: dict, history: dict, k: int = 5) -> list[dict]:
    """Drop-in for run_benchmark.py's ``retrieve()``: evidence from the ``k`` best sessions."""
    return HistoryIndex(history).evidence(question["question"], k_sessions=k)
