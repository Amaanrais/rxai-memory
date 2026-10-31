"""listen.py against a local stand-in for Reactor's chat socket: nothing sent, late answers heard.

    python -m unittest discover tests
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import listen  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402


class FakeChat:
    """A chat socket that refuses the first connection (busy), then sends ``replies`` after ``delay`` seconds and
    records whatever a client sends."""

    def __init__(self, replies: list[str], delay: float = 0.3):
        self.replies, self.delay, self.received, self.connections = replies, delay, [], 0
        self.auth: list[str | None] = []
        self.close_codes: list[int | None] = []  # how each listening connection was closed by the client
        self.loop = asyncio.new_event_loop()
        started = threading.Event()
        threading.Thread(target=self._run, args=(started,), daemon=True).start()
        started.wait(5)

    def _run(self, started: threading.Event) -> None:
        asyncio.set_event_loop(self.loop)

        async def handler(ws):
            self.connections += 1
            self.auth.append(ws.request.headers.get("Authorization"))
            if self.connections == 1:
                await ws.send(json.dumps({"type": "error", "code": "busy", "forwarded": False}))
                return
            sender = asyncio.ensure_future(self._send(ws))
            try:
                async for message in ws:
                    self.received.append(message)
            except ConnectionClosed as exc:
                self.close_codes.append(exc.rcvd.code if exc.rcvd else None)
            else:
                self.close_codes.append(ws.close_code)
            sender.cancel()

        async def main():
            async with serve(handler, "127.0.0.1", 0) as server:
                self.port = server.sockets[0].getsockname()[1]
                started.set()
                await asyncio.Future()

        self.loop.run_until_complete(main())

    async def _send(self, ws):
        await asyncio.sleep(self.delay)
        for n, text in enumerate(self.replies):
            await ws.send(json.dumps({"type": "reply", "text": text, "cycle": 10 + n}))


class ListenTest(unittest.TestCase):
    def setUp(self):
        os.environ["REACTOR_ACCESS_KEY"] = "test-key"
        listen.WINDOW = 1.0

    def test_hears_the_late_answer_without_sending(self):
        chat = FakeChat(["thinking...", "# FINAL ANSWER [ref:aaaa1111]: Hawaii [E1]",
                         "# FINAL ANSWER [ref:bbbb2222]: another question's answer"])
        started = time.monotonic()
        heard = listen.wait_for_late(f"http://127.0.0.1:{chat.port}", "agt_x", "R1", ["aaaa1111"], wait=10)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn({"ref": "aaaa1111", "answer": "Hawaii [E1]", "cycle": 11}, heard)
        self.assertEqual(chat.received, [])  # nothing sent
        self.assertEqual(chat.connections, 2)  # refused once (busy), then listened
        self.assertEqual(chat.auth[-1], "Bearer test-key")

    def test_gives_up_after_wait_with_nothing_heard(self):
        chat = FakeChat(["no final answer here"])
        started = time.monotonic()
        heard = listen.wait_for_late(f"http://127.0.0.1:{chat.port}", "agt_x", "R1", ["aaaa1111"], wait=2.5)
        self.assertEqual(heard, [])
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(chat.received, [])
        time.sleep(0.2)
        self.assertTrue(chat.close_codes and all(code == 1000 for code in chat.close_codes), chat.close_codes)


if __name__ == "__main__":
    unittest.main()
