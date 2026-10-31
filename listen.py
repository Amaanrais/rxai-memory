"""Listen on a Reactor chat without sending anything, for final answers that arrive late.

A client timeout doesn't cancel a question on Reactor's side: its answer can still come. wait_for_late() connects to
the same Reactor history the way reactor/client.py does, sends nothing, and collects every "# FINAL ANSWER [ref:...]"
it sees, until all the refs it waits for have answered or ``wait`` seconds have passed. A refusal or a closed chat is
retried with the client's backoff (1, 2, 4, 8 s). Answers come back in the client's late_replies format,
``{"ref", "answer", "cycle"}``. The URL and the final-answer parsing are reactor/client.py's own, imported unchanged.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path

import websockets

_spec = importlib.util.spec_from_file_location("reactor_client", Path(__file__).resolve().parent / "reactor" / "client.py")
reactor_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reactor_client)

WINDOW = 15.0  # seconds one connection listens before reconnecting
BACKOFF_FIRST, BACKOFF_MAX = 1.0, 8.0  # as the client's retries


async def _listen_once(url: str, key: str, refs: set[str], window: float) -> tuple[dict[str, dict], str | None]:
    """(final answers seen by ref, why the connection ended early or None), listening up to ``window`` seconds."""
    found: dict[str, dict] = {}
    end = time.monotonic() + window
    try:
        # each read times out on its own, so the connection always closes normally (code 1000), never cancelled
        async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                                      open_timeout=window) as ws:
            while (left := end - time.monotonic()) > 0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=left)
                except TimeoutError:
                    return found, None
                try:
                    frame = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(frame, dict):
                    continue
                if frame.get("type") == "error":
                    return found, str(frame.get("code") or "error")
                if frame.get("type") != "reply":
                    continue
                for ref, answer in reactor_client.final_answers(frame.get("text") or "").items():
                    found[ref] = {"ref": ref, "answer": answer, "cycle": frame.get("cycle")}
                if refs <= found.keys():
                    return found, None
        return found, None
    except TimeoutError:
        return found, "open timeout"
    except websockets.ConnectionClosed:
        return found, "closed"
    except websockets.InvalidStatus as exc:
        return found, f"HTTP {exc.response.status_code}"
    except OSError as exc:
        return found, type(exc).__name__
    return found, "closed"


def wait_for_late(base_url: str, agent_id: str, history: str, refs: list[str], wait: float = 120.0) -> list[dict]:
    """The final answers seen on Reactor history ``history`` within ``wait`` seconds, without sending anything:
    returns as soon as every ref in ``refs`` has one. Needs REACTOR_ACCESS_KEY in the environment."""
    key = os.environ["REACTOR_ACCESS_KEY"]
    url = reactor_client.chat_url(base_url, agent_id, history)
    deadline = time.monotonic() + wait
    found: dict[str, dict] = {}
    delay = BACKOFF_FIRST
    while (left := deadline - time.monotonic()) > 0:
        got, problem = asyncio.run(_listen_once(url, key, set(refs), min(WINDOW, left)))
        found.update(got)
        if set(refs) <= found.keys():
            break
        if problem:
            time.sleep(min(delay, max(deadline - time.monotonic(), 0)))
            delay = min(delay * 2, BACKOFF_MAX)
    return list(found.values())
