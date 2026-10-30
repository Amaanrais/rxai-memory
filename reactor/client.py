"""Ask your team's Reactor agent questions with the evidence your retrieval found, in one Reactor history per dataset
history; every run prints one JSON line.

Needs Python 3.11+ and ``pip install "websockets>=14"`` -- nothing else. Your access key comes from the environment,
never the command line (it would end up in your shell history):

    export REACTOR_ACCESS_KEY=rxk_...                  # Windows PowerShell: $env:REACTOR_ACCESS_KEY = "rxk_..."
    python client.py --base-url https://HOST --agent-id agt_... --new-history
        -> {"status": "history", "history_id": "9f2c...", ...}
    python client.py --base-url https://HOST --agent-id agt_... --history 9f2c... \\
        --question "When was my sister's wedding?" --question-date "2023/05/30 (Tue) 08:08" --evidence passages.json

**Histories.** ``--new-history`` starts a new, independent history: the agent's memory, chat history, pending input
and late replies are wiped (its configuration stays). Memory persists *within* a history -- ask its questions with
``--history ID`` -- never between two. A question to a history that has been replaced since is refused
(``history_changed``, with the agent's ``current_history_id``), never answered from the new one. ``--history`` is this
Reactor history's id (from ``--new-history``), not the benchmark's ``history_id`` (``h_...``): start one Reactor
history per benchmark history, ask all of that benchmark history's questions in it, and start a new one only for
the next benchmark history.

**One question at a time per agent.** The agent is stateful, so the server refuses rather than merges: a second open
chat (``busy``), a second question in one chat (``one_question``), and a question while the agent hasn't yet taken
up or replied to your previous one (``previous_pending``). Both ``busy`` and ``previous_pending`` are usual for a few
seconds right after an answer; the client asks again by itself, backing off 1-8 s within ``--timeout``, but only when
the server says the question never reached the agent (``"forwarded": false``). ``retries`` counts those. A question
over the server's own limit is refused (``too_large``, with ``limit``); that limit is the server's setting, not this
client's ``--max-chars``. For parallelism use several agents (one history at a time on each), if your team has them.

**Evidence.** ``--evidence`` is a UTF-8 file (a BOM is fine; ``-`` reads stdin -- on Windows, pass a file instead): a
JSON list of ``{"id": ..., "text": ...}`` objects (each with an optional ``"date"``, its session's date, passed on as
written) or of strings, or plain text with passages separated by blank lines (text that doesn't parse as JSON is plain
text, even when it starts with ``[``). The client sends one message: the question, the question date if given
(``--question-date``: ``2023-05-30``, ``2023-05-30 08:08`` or the benchmark's ``2023/05/30 (Tue) 08:08``; a time is
kept, wall-clock, with no timezone added), the passages labelled ``[E1]``..``[En]`` (with your ids and dates), a
request to cite them, and a fresh ref the agent is asked to put on its final answer line:
``# FINAL ANSWER [ref:<8 hex>]: ...`` -- or ``# FINAL ANSWER [ref:<8 hex>]: INSUFFICIENT_EVIDENCE`` when what it has
doesn't answer the question. The whole message may be at most ``--max-chars`` (default 16000), checked
before anything is sent. ``--text`` sends a text verbatim instead (no template, no ref, no final-answer check).

**Replies.** The agent answers asynchronously and may say other things first ("let me look"). After the ack (your
message queued, cycle N), only a reply from a later cycle can have seen your message, and only its
``# FINAL ANSWER [ref:<yours>]:`` segment is the answer -- a reply merely mentioning the ref is an interim note
(``other_replies``). A final answer for any *other* ref (an earlier question's, arriving late) is reported in
``late_replies``, never as this question's. The ref is text the model is asked to write, not a protocol id.

**Output** (stdout, one JSON line, ASCII -- other characters are ``\\uXXXX`` escapes, which any JSON parser turns
back into text). ``status``:
- ``answered`` (exit 0): ``answer`` (the final-answer segment), ``reply`` (its whole message), ``cycle``,
  ``latency_s``, ``citations`` (labels the answer cites, with your ids -- null where you gave none),
  ``unknown_citations`` (labels you never sent), ``uncited`` (evidence sent, none cited), ``abstained`` (the answer
  is ``INSUFFICIENT_EVIDENCE``), ``retries``;
- ``reply`` (exit 0, ``--text`` only): the first reply after the ack, *not* checked as a final answer;
- ``history`` (exit 0, ``--new-history``): ``history_id``, ``started_at``, ``running``, ``restarted``;
- ``timeout`` (exit 3), ``closed`` (exit 5: the agent stopped, a new history started, or access was revoked);
- ``refused`` (exit 4): ``code`` and ``text`` from the server (codes above, ``permission``, ``history_required``,
  ``history_resetting``, ``unsupported``), plus ``current_history_id`` (``history_changed``) or ``limit`` and
  ``size`` (``too_large``);
- ``error`` (exit 2): refused at connect (a new history refused too: ``busy``, ``still_stopping``,
  ``restart_failed``), bad input, or an unsupported Python/websockets.
Every outcome carries ``other_replies`` and ``late_replies`` where they apply, plus ``ref``, ``question``,
``question_date``, ``question_id`` and ``history_id`` when given. ``abstained`` and the citation fields are this client's own
reading of the reply; how they map onto the evaluator's fields is for the evaluator to confirm.

``--timeout`` (or ``REACTOR_TIMEOUT``) bounds the wait for the answer, retries included, in seconds. The default,
300, is provisional: the right value depends on measured answer latency and on how reliably the model writes the
final-answer line (without it, a question always runs to the timeout).
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from urllib.parse import quote, urlsplit

try:
    import websockets
except ImportError:  # environment_problem() says what to install
    websockets = None

MAX_CHARS = 16000
DEFAULT_TIMEOUT = 300.0  # provisional: replace once latencies are measured
# A final-answer line, "# FINAL ANSWER [ref:1a2b3c4d]: ...": like reactor-agent's own benchmark scorer
# (benchmark/scorers.py), flexible on '#'s, case, spacing, "_"/"-" and the colon, anchored at a line start.
_FINAL = re.compile(r"^[ \t]*#{1,3}[ \t]*final[ _-]?answer[ \t]*"
                    r"\[[ \t]*ref[ \t]*:[ \t]*([0-9a-f]{8})[ \t]*\][ \t]*:?[ \t]*", re.IGNORECASE | re.MULTILINE)
# --question-date: 2023-05-30, 2023-05-30 08:08, or the benchmark's 2023/05/30 (Tue) 08:08 (one separator throughout).
_DATE = re.compile(r"^(\d{4})([-/])(\d{2})\2(\d{2})(?: \(([A-Za-z]{3})\))?(?: (\d{2}):(\d{2}))?$")
# The agent's way of saying the evidence doesn't answer the question: the whole final answer, or its first word.
# A convention of this client (the evaluator's abstention field is yet to be confirmed), reported as "abstained".
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
_INSUFFICIENT = re.compile(rf"^\W*{INSUFFICIENT}\b", re.IGNORECASE)
# Refusals worth asking again, when the server says the question was never forwarded ("forwarded": false).
RETRYABLE = ("busy", "previous_pending")
_RETRY_FIRST, _RETRY_MAX = 1.0, 8.0  # seconds, doubling


def environment_problem(version: tuple = sys.version_info, ws=websockets) -> str | None:
    """Why this client can't run here, in words a participant can act on; None when it can. (asyncio.timeout is
    3.11+; websockets.connect takes additional_headers from 14 on.)"""

    if tuple(version[:2]) < (3, 11):
        return f"needs Python 3.11 or newer (this is {version[0]}.{version[1]})"
    if ws is None:
        return 'needs the websockets package: pip install "websockets>=14"'
    if int(ws.__version__.split(".")[0]) < 14:
        return f'needs websockets 14 or newer (this is {ws.__version__}): pip install -U "websockets>=14"'
    return None


class NoAnswer(Exception):
    """No answer to this request. ``other_replies``: replies after its ack that weren't it; ``late_replies``: final
    answers to other requests."""

    def __init__(self, message: str, other_replies: list[dict], late_replies: list[dict] | None = None) -> None:
        super().__init__(message)
        self.other_replies = other_replies
        self.late_replies = late_replies or []


class TimedOut(NoAnswer):
    pass


class Closed(NoAnswer):
    pass


class Refused(NoAnswer):
    """The server's ``error`` frame: ``code`` says why (module docstring)."""

    def __init__(self, code: str | None, text: str, other_replies: list[dict], late_replies: list[dict],
                 frame: dict | None = None) -> None:
        super().__init__(text or code or "refused by the server, no reason given", other_replies, late_replies)
        self.code = code
        frame = frame or {}
        # False only when the server says the question never reached the agent: then asking again is safe.
        self.forwarded = frame.get("forwarded")
        # history_changed: the agent's current history; too_large: the server's limit and this message's size.
        self.details = {k: frame[k] for k in ("history_id", "limit", "size") if k in frame}


def _base(base_url: str, ws: bool) -> str:
    """The server's base URL; plain http only to this machine -- anywhere else it would send the key in clear."""

    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return ("wss://" + base[len("https://"):]) if ws else base
    if base.startswith("http://") and urlsplit(base).hostname in ("127.0.0.1", "localhost"):
        return ("ws://" + base[len("http://"):]) if ws else base
    raise ValueError(f"{base_url!r}: use https:// (http:// only for 127.0.0.1/localhost)")


def chat_url(base_url: str, agent_id: str, history: str | None = None) -> str:
    url = f"{_base(base_url, ws=True)}/agents/{agent_id}/ws/chat"
    return f"{url}?history={quote(history, safe='')}" if history else url


def parse_question_date(value: str) -> str:
    """``YYYY-MM-DD``, or ``YYYY-MM-DD HH:MM`` when the value has a time -- the benchmark's own
    ``2023/05/30 (Tue) 08:08`` included, its weekday checked against the date. A real date and time, else ValueError.
    No timezone is added: the value is wall-clock time, as given."""

    m = _DATE.match(value or "")
    if not m:
        raise ValueError(f"--question-date {value!r}: use YYYY-MM-DD, YYYY-MM-DD HH:MM or 2023/05/30 (Tue) 08:08")
    year, month, day, weekday, hour, minute = m.group(1, 3, 4, 5, 6, 7)
    try:
        when = datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0))
    except ValueError as exc:
        raise ValueError(f"--question-date {value!r}: not a calendar date and time") from exc
    if weekday and weekday.title() != when.strftime("%a"):
        raise ValueError(f"--question-date {value!r}: {when:%Y-%m-%d} is a {when:%a}, not {weekday}")
    return when.strftime("%Y-%m-%d %H:%M" if hour else "%Y-%m-%d")


def load_evidence(raw: str) -> list[dict]:
    """Passages as ``{"id", "text"}`` (``id`` None when not given), plus ``"date"`` when given: a JSON list of strings
    or ``{"id", "text"}`` objects, each with an optional ``"date"`` (the session's date, a one-line string passed on
    as written), else plain text split on blank lines. Only what parses as JSON is held to that shape -- a chat log
    whose first line is ``[2024-05-01 18:02] ...`` is plain text."""

    try:
        items = json.loads(raw) if raw.lstrip().startswith("[") else None
    except ValueError:
        items = None
    if items is not None:
        passages = []
        for item in items if isinstance(items, list) else [None]:
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError('evidence: a JSON list of strings or of {"id", "text"} objects, each with text')
            pid, when = item.get("id"), item.get("date")
            if when is not None and (not isinstance(when, str) or "\n" in when):
                raise ValueError('evidence: a passage\'s "date" is a one-line string, e.g. "2023/05/20 (Sat) 02:21"')
            passage = {"id": None if pid is None else str(pid), "text": item["text"].strip()}
            if when and when.strip():
                passage["date"] = when.strip()
            passages.append(passage)
    else:
        passages = [{"id": None, "text": block.strip()} for block in re.split(r"\n\s*\n", raw) if block.strip()]
    if not passages:
        raise ValueError("evidence: no passages")
    return passages


def build_message(question: str, passages: list[dict], ref: str, question_date: str | None = None) -> str:
    """The one message for a question, its date and its passages, asking for ``# FINAL ANSWER [ref:<ref>]:``."""

    question = question.strip()
    if not question:
        raise ValueError("the question is empty")
    lines = [f"[ref:{ref}] Question: {question}"]
    if question_date and " " in question_date:  # with a time: wall-clock, no timezone (none is given)
        day, hhmm = question_date.split(" ")
        weekday = datetime.strptime(day, "%Y-%m-%d").strftime("%a")
        lines.append(f"Question date: {day} ({weekday}) {hhmm}, no timezone. Answer as of that moment: treat it as now "
                     "for this question; the clock you see shows the real date and time -- ignore it here.")
    elif question_date:
        lines.append(f"Question date: {question_date}. Answer as of that date: treat {question_date} as today for "
                     "this question; the clock you see shows the real date -- ignore it here.")
    if passages:
        lines += ["", "Evidence:"]
        for n, passage in enumerate(passages, 1):
            about = [f"source: {passage['id']}"] if passage["id"] else []
            about += [f"date: {passage['date']}"] if passage.get("date") else []
            source = f" ({', '.join(about)})" if about else ""
            lines.append(f"[E{n}]{source} {passage['text']}")
        lines += ["", "Answer from the evidence above and cite each passage you use by its label, e.g. [E1]."]
    lines += ["", "When you have your answer, write it on a line of its own that starts exactly with:",
              f"# FINAL ANSWER [ref:{ref}]: ",
              f"If the information you have doesn't answer the question, that line is: # FINAL ANSWER [ref:{ref}]: "
              f"{INSUFFICIENT}",
              "Anything you send before that line is an interim note, not the answer."]
    return "\n".join(lines)


def final_answers(reply: str) -> dict[str, str]:
    """Each ``# FINAL ANSWER [ref:...]:`` segment of ``reply`` by ref -- the text after the marker, up to the next
    marker or the end (a combined reply answering two questions splits); a ref marked twice keeps its last."""

    marks = list(_FINAL.finditer(reply))
    found: dict[str, str] = {}
    for n, mark in enumerate(marks):
        end = marks[n + 1].start() if n + 1 < len(marks) else len(reply)
        found[mark.group(1).lower()] = reply[mark.end():end].strip()
    return found


def citations(answer: str, passages: list[dict]) -> tuple[list[dict], list[str]]:
    """``(cited, unknown)``: the ``[E#]`` labels in ``answer`` (``[E1, E3]`` too) that were sent, with their ids as
    given (None where none was), and those that weren't."""

    groups = re.findall(r"\[([^\[\]]*)\]", answer)
    cited = sorted({int(n) for group in groups for n in re.findall(r"\bE(\d+)\b", group)})
    known = [{"label": f"E{n}", "id": passages[n - 1]["id"]} for n in cited if 1 <= n <= len(passages)]
    return known, [f"E{n}" for n in cited if not 1 <= n <= len(passages)]


async def ask(base_url: str, agent_id: str, access_key: str, text: str, *, timeout: float,
              ref: str | None = None, history: str | None = None) -> dict:
    """Sends ``text`` and waits for ``ref``'s final answer (module docstring); with no ``ref`` (``--text``), for the
    first reply after the ack. Returns ``{"answer"?, "reply", "cycle", "latency_s", "other_replies",
    "late_replies"}``; raises TimedOut, Closed or Refused (each with what arrived), or ConnectionRefusedError."""

    started = time.monotonic()
    url = chat_url(base_url, agent_id, history)
    headers = {"Authorization": f"Bearer {access_key}"}
    others: list[dict] = []
    late: list[dict] = []
    try:
        async with asyncio.timeout(timeout):
            async with websockets.connect(url, additional_headers=headers) as ws:
                try:
                    await ws.send(json.dumps({"type": "message", "text": text}))
                except websockets.ConnectionClosed:
                    # The server may refuse at connect (busy, history_*) and close while this was being sent: the
                    # refusal frame is already buffered, read below. Without one, this ends as Closed -- never retried.
                    pass
                ack_cycle = None
                async for raw in ws:
                    try:
                        frame = json.loads(raw)
                    except ValueError:
                        continue  # not a protocol frame: never interpreted
                    if not isinstance(frame, dict):
                        continue
                    kind = frame.get("type")
                    if kind == "error":  # the code as the server gave it (None if it gave none): never invented
                        code = frame.get("code")
                        raise Refused(code if isinstance(code, str) else None, str(frame.get("text") or ""), others,
                                      late, frame)
                    if kind == "ack" and ack_cycle is None and frame.get("text") == text:
                        ack_cycle = frame.get("cycle", 0)
                        continue
                    if kind != "reply":
                        continue
                    reply = frame.get("text") or ""
                    cycle = frame.get("cycle")
                    finals = final_answers(reply)
                    late += [{"ref": r, "answer": a, "cycle": cycle} for r, a in finals.items() if r != ref]
                    if ack_cycle is None or not isinstance(cycle, int) or cycle <= ack_cycle:
                        continue  # queued from before, or from a cycle that took its input before this message
                    result = {"reply": reply, "cycle": cycle, "latency_s": round(time.monotonic() - started, 2),
                              "other_replies": others, "late_replies": late}
                    if ref is None:
                        return result
                    if ref in finals:
                        return {"answer": finals[ref], **result}
                    others.append({"text": reply, "cycle": cycle})
    except TimeoutError as exc:
        raise TimedOut(f"no final answer within {timeout:g}s", others, late) from exc
    except websockets.InvalidStatus as exc:
        status = exc.response.status_code
        why = ("wrong key, no access to this agent, or the agent isn't running" if status in (401, 403)
               else "no chat at that address: --base-url is just https://HOST, no path" if status == 404
               else "is --base-url the Reactor server?")
        raise ConnectionRefusedError(f"refused at connect (HTTP {status}): {why}") from exc
    except websockets.ConnectionClosed as exc:
        reason = exc.rcvd.reason if exc.rcvd else ""
        raise Closed(f"closed by the server ({exc.rcvd.code if exc.rcvd else '?'}): {reason}", others, late) from exc
    raise Closed("the server closed the chat before an answer", others, late)


def ask_retrying(base_url: str, agent_id: str, access_key: str, text: str, *, timeout: float,
                 ref: str | None = None, history: str | None = None) -> tuple[dict, int]:
    """ask(), asked again after a ``busy`` or ``previous_pending`` refusal that the server says never reached the agent
    (``"forwarded": false``) -- backing off 1, 2, 4, 8 s, all within ``timeout``. Late replies seen along the way
    are kept. Returns ``(result, retries)``; what it raises carries ``retries``."""

    deadline = time.monotonic() + timeout
    delay, retries, late = _RETRY_FIRST, 0, []
    while True:
        try:
            result = asyncio.run(ask(base_url, agent_id, access_key, text, ref=ref, history=history,
                                     timeout=max(deadline - time.monotonic(), 0.001)))
        except NoAnswer as exc:
            exc.late_replies = late + exc.late_replies
            exc.retries = retries
            if (isinstance(exc, Refused) and exc.code in RETRYABLE and exc.forwarded is False
                    and deadline - time.monotonic() > delay):
                late = exc.late_replies
                time.sleep(delay)
                delay, retries = min(delay * 2, _RETRY_MAX), retries + 1
                continue
            raise
        result["late_replies"] = late + result["late_replies"]
        return result, retries


def new_history(base_url: str, agent_id: str, access_key: str, *, timeout: float) -> dict:
    """POST /agents/{agent_id}/history: a new, independent history (module docstring)."""

    req = urllib.request.Request(f"{_base(base_url, ws=False)}/agents/{agent_id}/history", method="POST", data=b"",
                                 headers={"Authorization": f"Bearer {access_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read() or b"{}").get("detail")
        except ValueError:
            detail = None
        raise ConnectionRefusedError(f"new history refused (HTTP {exc.code}): {detail or exc.reason}") from exc


def _read(path: str) -> str:
    """The evidence as text: UTF-8, with or without a BOM, or UTF-16 with its BOM (what Windows PowerShell 5.1's
    ``Out-File`` and ``>`` write)."""

    try:
        if path == "-":
            raw = sys.stdin.buffer.read()
        else:
            with open(path, "rb") as fh:
                raw = fh.read()
        if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return raw.decode("utf-16")
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"evidence: not UTF-8 text ({exc.reason} at byte {exc.start}): save it as UTF-8") from exc


def _timeout(value: str | None) -> float:
    raw = value if value is not None else os.environ.get("REACTOR_TIMEOUT")
    if raw is None or raw == "":
        return DEFAULT_TIMEOUT
    try:
        seconds = float(raw)
    except ValueError:
        seconds = -1.0
    if not 0 < seconds <= 86400:
        raise ValueError(f"timeout {raw!r}: seconds, more than 0")
    return seconds


def _emit(code: int, out: dict, message: str | None = None) -> int:
    if message:
        print(f"error: {message}", file=sys.stderr)
    print(json.dumps(out))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask your Reactor agent a question and print its final answer.")
    parser.add_argument("--base-url", required=True, help="the organizers' https://HOST, no path")
    parser.add_argument("--agent-id", required=True, help="your team's agent id (agt_...)")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--new-history", action="store_true", help="start a new, independent history")
    what.add_argument("--question", help="your question, sent with --evidence in the template")
    what.add_argument("--text", help="send this text verbatim instead: no template, no final-answer check")
    parser.add_argument("--history", help="the history this question belongs to (from --new-history)")
    parser.add_argument("--question-date", help="when the question is asked: YYYY-MM-DD, YYYY-MM-DD HH:MM, or the "
                                                "benchmark's '2023/05/30 (Tue) 08:08' (no timezone)")
    parser.add_argument("--question-id", help="the benchmark's id for this question: only echoed in the output "
                                              "(for predictions.py), never sent")
    parser.add_argument("--evidence", help="retrieved passages: a JSON list or blank-line-separated text; - for stdin")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS,
                        help=f"refuse a longer message, before sending (default {MAX_CHARS})")
    parser.add_argument("--timeout", help=f"seconds to wait (default: REACTOR_TIMEOUT, else {DEFAULT_TIMEOUT:g}; "
                                          "provisional)")
    args = parser.parse_args(argv)
    problem = environment_problem()
    if problem:
        return _emit(2, {"status": "error", "error": problem}, problem)
    key = os.environ.get("REACTOR_ACCESS_KEY", "")
    if not key:
        return _emit(2, {"status": "error", "error": "set REACTOR_ACCESS_KEY"}, "set REACTOR_ACCESS_KEY")
    out: dict = {}
    retries = 0
    try:
        timeout = _timeout(args.timeout)
        if args.new_history:
            if args.question_date or args.evidence is not None or args.history:
                raise ValueError("--new-history takes no --history, --question-date or --evidence")
            result = new_history(args.base_url, args.agent_id, key, timeout=timeout)
            return _emit(0, {"status": "history", **result})
        if args.question_id:
            out["question_id"] = args.question_id
        if args.history:
            out["history_id"] = args.history
        if args.text is not None:
            if args.evidence is not None or args.question_date:
                raise ValueError("--evidence and --question-date go with --question, not --text")
            text, ref, passages = args.text, None, []
            out["text"] = text
        else:
            question_date = parse_question_date(args.question_date) if args.question_date else None
            passages = load_evidence(_read(args.evidence)) if args.evidence is not None else []
            ref = secrets.token_hex(4)
            text = build_message(args.question, passages, ref, question_date)
            out.update(ref=ref, question=args.question, question_date=question_date)
        if len(text) > args.max_chars:
            raise ValueError(f"the message is {len(text)} characters, over --max-chars {args.max_chars}: "
                             "send fewer or shorter passages")
        result, retries = ask_retrying(args.base_url, args.agent_id, key, text, timeout=timeout, ref=ref,
                                       history=args.history)
    except Refused as exc:
        details = dict(exc.details)
        if "history_id" in details:  # never confuse it with the history this question named (out["history_id"])
            details["current_history_id"] = details.pop("history_id")
        return _emit(4, {"status": "refused", **out, "code": exc.code, "text": str(exc), **details,
                         "retries": getattr(exc, "retries", 0),
                         "other_replies": exc.other_replies, "late_replies": exc.late_replies}, f"refused: {exc}")
    except NoAnswer as exc:
        status, code = ("timeout", 3) if isinstance(exc, TimedOut) else ("closed", 5)
        return _emit(code, {"status": status, **out, "other_replies": exc.other_replies,
                            "late_replies": exc.late_replies, "retries": getattr(exc, "retries", 0)}, str(exc))
    except (OSError, ValueError) as exc:
        return _emit(2, {"status": "error", **out, "error": str(exc)}, str(exc))
    if ref is None:
        return _emit(0, {"status": "reply", **out, **result})
    cited, unknown = citations(result["answer"], passages)
    result["retries"] = retries
    return _emit(0, {"status": "answered", **out, **result, "citations": cited, "unknown_citations": unknown,
                     "uncited": bool(passages) and not cited,
                     "abstained": bool(_INSUFFICIENT.match(result["answer"]))})


if __name__ == "__main__":
    sys.exit(main())
