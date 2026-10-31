"""Builds recall.mp4: the deck and the Recall app, screen-recorded, with a neural voiceover.

    streamlit run app.py --server.address localhost --server.port 8501     # in another terminal
    uv run --with playwright python presentation/video/make_video.py

Needs ffmpeg and uv on PATH (edge-tts, run through uvx, sends the narration text to Microsoft's TTS service).
Edit SEGMENTS to change what is said or shown; every run rebuilds everything from scratch.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
DECK = (HERE.parent / "index.html").as_uri()
APP = "http://localhost:8501"
OUT = HERE / "recall.mp4"
WORK = HERE / "build"
VOICE, RATE = "en-US-AndrewMultilingualNeural", "+4%"
SIZE = {"width": 1920, "height": 1080}
GAP = 0.7  # seconds of quiet after each line

# (what to show, narration). ("slide", n) is deck slide n; ("app", action) is a step in the Recall app.
SEGMENTS = [
    (("slide", 1), "This is Recall, our solution to the RxAI Memory Retrieval Challenge, by Amaan Rais, Dedar Alam "
                   "and Frensi Qatipi. Recall finds the few turns that matter in months of chat history, and cites "
                   "only what supports the answer."),
    (("slide", 2), "Here's the problem. Each test user has two hundred conversations: about one point six six million "
                   "characters. Reactor takes sixteen thousand characters per message. Some questions can't be "
                   "answered at all, and near-miss conversations are planted to look relevant."),
    (("slide", 4), "So our code picks the evidence, and Reactor answers. We index every turn, route the question by "
                   "type, retrieve the best turns with their dates, let Reactor answer from labelled excerpts, then "
                   "check its citations, or abstain."),
    (("slide", 5), "A few decisions matter most. We search turns, not whole conversations, because a fact usually "
                   "lives in one turn. Counting and ordering questions get a bigger budget. And dates are worked out "
                   "in our code: every excerpt says how many days before the question it was said. All in plain "
                   "Python, with no external models."),
    (("slide", 6), "Citations follow the organisers' rule: they must support the answer. Reactor's own labels come "
                   "first, then excerpts that actually contain its answer, and never the top result by default. When "
                   "Reactor says the evidence is insufficient, we abstain, with no citations."),
    (("app", "open"), "This is our demo app. We pick a question: what did I bake for my niece's birthday party? "
                      "Recall searches this user's whole chat history, and reads about forty times less than all of "
                      "it."),
    (("app", "sources"), "Under Sources is exactly what Reactor would receive: a few dated excerpts, each saying how "
                         "many days before the question it happened."),
    (("app", "update"), "For a fact that changed, like a frequent flyer status, the latest excerpt is marked most "
                        "recent, so the newest value wins, unless the question asks for the previous one."),
    (("app", "results"), "And the Results tab shows our measured retrieval numbers, straight from the evaluation "
                         "files."),
    (("slide", 7), "That focus pays off. We send a median of eight point two thousand characters per question: about "
                   "two hundred times less than the full history, and nineteen to twenty-three times less per user "
                   "than loading the whole history into Reactor."),
    (("slide", 8), "By question type, a correct conversation is in our top five for over ninety-five percent of "
                   "questions in every type. The hard part is multi-session and temporal questions, where the answer "
                   "is spread over several turns."),
    (("slide", 9), "Overall, a correct conversation is in the top five for ninety-eight point two percent of dev "
                   "questions, and ninety-three point five percent on dev-large, with five hundred conversations per "
                   "user. Test scores come from the organisers."),
    (("slide", 10), "We also built and measured version two: focused trimming, more room for a clear leader, time "
                    "sentences, and a missing-words hint against traps. It's ready for the next run."),
    (("slide", 11), "And we played by the rules: every answer comes from Reactor, one Reactor history per user, "
                    "careful timeout handling, and evidence logs that make every prediction reproducible."),
    (("slide", 13), "Next: a local reranker for questions worded differently from the chat, dates computed in code, "
                    "and smarter abstention."),
    (("slide", 14), "Thank you. That's Recall: the needle, not the haystack."),
]


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def duration(path: Path) -> float:
    return float(run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)))


def voice() -> list[float]:
    """One mp3 per segment; returns each segment's length on screen (speech + GAP)."""
    lengths = []
    for i, (_, text) in enumerate(SEGMENTS):
        mp3 = WORK / f"v{i:02}.mp3"
        run("uvx", "--from", "edge-tts>=7.3.1", "edge-tts", "--voice", VOICE, f"--rate={RATE}", "--text", text, "--write-media", str(mp3))
        lengths.append(duration(mp3) + GAP)
    return lengths


def scroll(page, steps: int, dy: int, pause: int = 700) -> None:
    for _ in range(steps):
        page.mouse.wheel(0, dy)
        page.wait_for_timeout(pause)


def app_step(page, action: str) -> None:
    page.mouse.move(640, 420)  # wheel events go to what's under the pointer: the main column
    if action == "open":
        page.get_by_role("button", name="A fact the user mentioned").click()
    elif action == "sources":
        page.get_by_role("tab", name="Sources").click()
        scroll(page, 6, 260)
    elif action == "update":
        scroll(page, 1, -6000, 400)
        page.get_by_role("button", name="A fact that changed").click()
        page.wait_for_timeout(2500)
        scroll(page, 5, 240)
    elif action == "results":
        scroll(page, 1, -6000, 400)
        page.get_by_role("tab", name="Results").click()
        page.wait_for_timeout(1500)
        scroll(page, 4, 300, 900)


def record(blocks: list[list[int]], lengths: list[float]) -> list[Path]:
    """One recording per run of consecutive deck or app segments; returns clips trimmed to their segments."""
    clips = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for b, block in enumerate(blocks):
            kind = SEGMENTS[block[0]][0][0]
            # the app's narrow column is unreadable at 1920 px: record it at 1280 px, scaled up when encoding
            view = SIZE if kind == "slide" else {"width": 1280, "height": 720}
            ctx = browser.new_context(viewport=view, record_video_dir=str(WORK), record_video_size=view)
            page = ctx.new_page()
            opened = time.monotonic()
            if kind == "slide":
                page.goto(DECK)
                page.evaluate(f"go({SEGMENTS[block[0]][0][1] - 1})")
            else:
                page.goto(APP)
                page.get_by_role("button", name="A fact the user mentioned").wait_for(timeout=120_000)
            page.wait_for_timeout(1500)  # let the first view settle
            start = due = time.monotonic()
            for i in block:
                target = SEGMENTS[i][0][1]
                if kind == "slide":
                    page.evaluate(f"go({target - 1})")
                else:
                    app_step(page, target)
                due += lengths[i]
                page.wait_for_timeout(max(0, (due - time.monotonic()) * 1000))
            video = page.video.path()
            ctx.close()
            clip = WORK / f"clip{b:02}.mp4"
            run("ffmpeg", "-y", "-ss", f"{start - opened:.3f}", "-i", str(video), "-t", f"{due - start:.3f}",
                "-vf", f"scale={SIZE['width']}:{SIZE['height']}:flags=lanczos,fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "18", "-an", str(clip))
            clips.append(clip)
    return clips


def main() -> None:
    WORK.mkdir(exist_ok=True)
    lengths = voice()
    blocks: list[list[int]] = []  # runs of consecutive segments of the same kind
    for i, ((kind, _), _) in enumerate(SEGMENTS):
        if blocks and SEGMENTS[blocks[-1][-1]][0][0] == kind:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    clips = record(blocks, lengths)

    # video: the clips back to back; audio: each line padded to its segment's length
    (WORK / "clips.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(WORK / "clips.txt"), "-c", "copy",
        str(WORK / "video.mp4"))
    n = len(SEGMENTS)
    inputs = [arg for i in range(n) for arg in ("-i", str(WORK / f"v{i:02}.mp3"))]
    pads = "".join(f"[{i}]apad=whole_dur={lengths[i]:.3f}[a{i}];" for i in range(n))
    joined = "".join(f"[a{i}]" for i in range(n))
    run("ffmpeg", "-y", *inputs, "-filter_complex", f"{pads}{joined}concat=n={n}:v=0:a=1[a]",
        "-map", "[a]", str(WORK / "voice.wav"))
    total = sum(lengths)
    run("ffmpeg", "-y", "-i", str(WORK / "video.mp4"), "-i", str(WORK / "voice.wav"),
        "-vf", f"fade=in:0:15,fade=out:st={total - 0.6:.3f}:d=0.6", "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(OUT))
    print(json.dumps({"out": str(OUT), "seconds": round(total, 1), "segments": n}))


if __name__ == "__main__":
    main()
