# Recall: grounded long-term memory on Reactor

Our submission to the RxAI Memory Retrieval Challenge. The organisers' brief is in [CHALLENGE.md](CHALLENGE.md);
the data format is in [format.md](format.md).

## 1. Summary

For each question, our own code finds the few turns that matter in a history of hundreds of conversations
(turn-level BM25, standard library only), sends them to Reactor as dated, labelled excerpts, and turns Reactor's
reply into a prediction. **Every final answer comes from Reactor.** Citations are Reactor's own `[E#]` labels; when
it gives none, we cite only excerpts that actually contain its answer. When Reactor says the evidence doesn't answer
the question, we abstain, with no citations.

Headline results:

| | |
|---|---|
| Retrieval, dev (526 questions, ~50 sessions per history) | a gold session in the top 5 for **98.2%** of answerable questions |
| Retrieval, dev-large (130 questions, ~500 sessions per history) | **93.5%** |
| Characters sent per test question | median **8.2k** (whole message) vs **~1.66M** for the full history |
| Test answers | placeholder: see [Results](#9-results) |

## 2. Architecture

```mermaid
flowchart LR
    H[(histories.jsonl)] --> I["1 · Index<br/>every turn, both roles,<br/>BM25 per history"]
    Q[question + date] --> R["2 · Route<br/>count / order / temporal:<br/>8 sessions, 24k chars<br/>else 5 sessions, 12k"]
    I --> T
    R --> T["3 · Retrieve and trim<br/>best turns ± 1 per session,<br/>dates, N days before,<br/>[most recent]"]
    T --> X["4 · Reactor answers<br/>from excerpts [E1]..[En],<br/>labels on the answer line"]
    X --> C["5 · Citation check / abstain<br/>labels → session ids,<br/>else support check;<br/>INSUFFICIENT_EVIDENCE → abstain"]
    C --> P[(predictions.jsonl)]
```

| File | Role |
|---|---|
| `retrieval.py` | the index, routing, evidence building (`HistoryIndex`) |
| `run.py` | the benchmark driver: one Reactor history per dataset history, via `reactor/client.py` |
| `make_predictions.py` | results → `predictions.jsonl`: citation rules, abstention, late-reply recovery |
| `eval_retrieval.py` | offline retrieval metrics on dev splits (no Reactor calls) |
| `run_v2.py` | the v2 driver (section 7), not used for the test predictions |
| `app.py` | the Recall demo (section 11) |

## 3. Setup

```sh
python3 --version                          # 3.11 or newer (reactor/client.py needs it)
python3 -m venv .venv && . .venv/bin/activate
pip install websockets streamlit           # websockets for the Reactor client; streamlit only for the demo
```

Data: download `dev.tar.gz` and `test.tar.gz` (and dev-large, optional) from the challenge's
[Releases](https://github.com/RxAI-dev/memory-challenge/releases) page and extract them in the repo root, giving
`dev/`, `test/` (and `dev-large/`). `convert_longmemeval.py` rebuilds the dev splits from LongMemEval if needed.

Access key: type it at a hidden prompt, in the terminal you run from. **Never put it on a command line, in a file
or in a chat.**

```sh
printf 'Access key: '; read -rs REACTOR_ACCESS_KEY; echo; export REACTOR_ACCESS_KEY
```

## 4. Reproducing predictions.jsonl

```sh
python run.py --questions test/questions.jsonl --histories test/histories.jsonl \
    --base-url https://tNN.api.reactor-hackathon.rxai.dev --agent-id agt_... --out results-test.jsonl
python make_predictions.py results-test.jsonl --questions test/questions.jsonl --out predictions.jsonl
```

`run.py` appends each answer to `results-test.jsonl` as it arrives and logs the evidence it sent to
`results-test.evidence.jsonl`; `make_predictions.py` validates its output with `score.py`'s format check.
On dev, `python score.py dev/gold.jsonl predictions.jsonl` scores it.

**Which code produced the test predictions.**

| Result lines | Driver | Notes |
|---|---|---|
| 1–32 | tag `v1-test-run` | client timeout 300 s (the kit's default) |
| 33 onwards | tag `v1.1-test-run` | after a mid-run restart: driver changes only (section 5) |

`predictions.jsonl` is built with `make_predictions.py` at `v1.1-test-run`. Between the two tags, **v1's evidence
and question text are byte-identical**: we ran both drivers on all 526 dev questions with a recording stand-in for
the client and compared the question text, date, `--max-chars` and evidence bytes of every question (0
differences), and the evidence logs (identical, 9,050,462 bytes). Only the driver's behaviour changed.

## 5. Driver behaviour

Following the organisers' guidance on Discord:

- **One Reactor history per dataset history**, started with `--new-history`, never reset mid-history.
- **`--timeout 600`** for every question.
- **Timeouts are retried once in the same Reactor history**, 30 s later; the client keeps retrying while the
  previous request hasn't cleared (`previous_pending`).
- **Late answers are recovered.** If Reactor's final answer to a timed-out question arrives later, in another
  question's `late_replies` (matched by its ref), it is recorded as that question's answer, marked `recovered`.
- **Resume without a reset.** After a restart, the dataset history in progress continues in its Reactor history
  (the agent's current one), before any other history is started.

Transparency: 2 questions (`q_1a6b4fa2ede3`, `q_aa27ddd3b7c5`) timed out under `v1-test-run`, and no late
answer for them ever arrived. Their Reactor history had already been replaced when we restarted, so they are
asked again in a fresh Reactor history for their dataset history, the kit's documented rerun behaviour
(`reactor/QUICKSTART.md`, "A rerun starts a new Reactor history"). Every question carries its own evidence, so
nothing from the earlier history is needed.

## 6. Design decisions

- **Turn-level BM25 with session aggregation.** Each turn is a chunk (long turns split into 1,500-character
  windows); a session scores its best chunk plus 0.25 × its next two. Turns, not sessions, because a session is
  ~10k characters and the fact is usually in one turn.
- **Assistant turns are indexed.** "What did you recommend…" questions are answered only by assistant turns.
- **Question routing.** Counting, totalling, comparing and ordering questions ("how many", "which first", "how
  long ago") get 8 sessions in 24,000 characters (`--max-chars 30000`); others 5 sessions in 12,000.
- **Dates done in code.** Each excerpt carries its session date and "(N days before the question date)"; the
  latest excerpt is marked `[most recent]`, so knowledge-update questions take the latest value.
- **Answer instructions.** Every question: put the `[E#]` labels on the FINAL ANSWER line, one value, no
  hedging. Questions about a value that may have changed: "the most recent date wins", except when the question
  asks for the previous or original value.
- **Citation rules** (per the organisers' ruling that fallback citations must support Reactor's answer):
  1. Reactor's labels on the answer line;
  2. else the labels in the last paragraph of its reasoning;
  3. else a support check: the excerpts that contain Reactor's answer (normalized as `score.py` does), or ≥ 60%
     of its content words, and only if that picks out at most half of the excerpts (a bare "2" matches all of
     them, so it cites nothing);
  4. abstentions cite nothing. We never fall back to the top-ranked session.

## 7. v2: built and measured, not used for the test predictions

v2 (`retrieval.py` `version=2`, `run_v2.py`) was built and measured on dev, but there wasn't the Reactor time to
run it on test. It adds:

- **Focused trimming**: a long turn keeps the stretch with the most question words, not its beginning.
- **Lead-session budget**: a clearly leading session gets up to half the budget.
- **Time sentences**: for temporal and counting questions, the user's sentences that say when something happened
  ("last month", "three weeks ago"), from anywhere in a sent session.
- **Missing-words hint** for near-miss traps: "the evidence never mentions: X. If the answer depends on these,
  answer INSUFFICIENT_EVIDENCE".
- **Work-it-out prompt** for aggregate questions: compute a difference, total, count or order before the
  answer line.

| | hit@5 v1 | hit@5 v2 | sent v1 | sent v2 | ans∈ev v1 | ans∈ev v2 |
|---|---|---|---|---|---|---|
| dev (496 answerable) | 0.982 | 0.982 | 0.954 | 0.954 | 0.542 | **0.556** |
| dev-large (124 answerable of 130) | 0.935 | 0.935 | 0.878 | 0.878 | 0.484 | **0.492** |

*hit@5*: a gold session in the top 5; *sent*: share of gold sessions in the evidence sent; *ans∈ev*: the gold
answer appears in the evidence sent. A tried change, dropping a session's weaker second hit, lowered ans∈ev and
was left out.

The missing-words hint fires on **15 of 30 (50%)** unanswerable dev questions, against **62 of 496 (12.5%)**
answerable ones; on dev-large, 3 of 6 against 2 of 124 (1.6%).

## 8. Compliance

- Final answers come only from Reactor, through the organisers' `reactor/client.py`; we don't edit them.
- No external LLM or embedding APIs anywhere in the pipeline.
- Standard library only, apart from `websockets` (the Reactor client) and `streamlit` (the demo).

## 9. Results

**Retrieval** (offline, `eval_retrieval.py`; v1, the version used on test):

| dev, by type | n | hit@5 | sent | ans∈ev |
|---|---|---|---|---|
| knowledge-update | 72 | 1.000 | 0.993 | 0.819 |
| multi-session | 121 | 0.983 | 0.913 | 0.430 |
| single-session-assistant | 112 | 1.000 | 1.000 | 0.554 |
| single-session-user | 64 | 0.984 | 0.984 | 0.875 |
| temporal-reasoning | 127 | 0.953 | 0.916 | 0.315 |
| **all** | 496 | **0.982** | **0.954** | **0.542** |

On dev-large (~500 sessions per history, 130 questions): hit@5 0.935, sent 0.878.

**Reactor on a dev sample** (`sample10`: 2 questions per type plus 1 unanswerable; `score.py`):

| type | n | correct | unjudged | abstention recall | false abstention | evidence F1 |
|---|---|---|---|---|---|---|
| knowledge-update | 2 | 0.500 | 0.500 | – | 0.000 | 0.667 |
| multi-session | 3 | 0.333 | 0.333 | 0.000 | 0.000 | 0.733 |
| single-session-assistant | 2 | 0.000 | 0.500 | – | 0.500 | 0.333 |
| single-session-user | 2 | 1.000 | 0.000 | – | 0.000 | 1.000 |
| temporal-reasoning | 2 | 0.000 | 1.000 | – | 0.000 | 0.667 |
| **all** | 11 | **0.364** | 0.455 | 0.000 | 0.100 | **0.680** |

"Unjudged" answers aren't exact matches and go to the LLM judge in official scoring. A sample of 11 is small; we
diagnosed its failures by hand (an answer cut off by the excerpt budget, a near-miss trap answered instead of
abstaining, a two-session difference, an event order needing a date from elsewhere in a session), and they
motivated v2.

**Test**: _placeholder, to be filled in from the official scores._

## 10. Efficiency

- **Per question.** The whole message we send (question, instructions, excerpts, the client's template) has a
  median of **8.2k characters** (max 14.1k) over the first 37 test questions, against **~1.66M characters** for
  a full test history: roughly 200× less per question.
- **Per dataset history**, against the organisers' baseline of loading the whole history once: a test history
  has 9–10 questions, so we send ~10 × 8.5k ≈ 85k characters against ~1.66M, **19–23× less** as measured. Even if
  every question used its full budget, the ratio would stay above **~6.5×** (all routed, 24k) to **~12×** (none
  routed). Loading also takes 65–125 messages per history (at 32k or 16k characters each), each a full agent turn.

## 11. Demo app: Recall

`app.py` shows the pipeline on any dev or test question, or on your own question about a history: Reactor's answer
(with an *Abstained* badge), the gold answer and a correct / incorrect / needs-judge marker on dev, the evidence
timeline (dates, days before, *most recent*, speakers, cited excerpts highlighted, the v2 missing-words hint),
characters and tokens sent vs the full history, a "How it works" walkthrough, and a Results tab.

```sh
streamlit run app.py --server.address localhost
```

It reads cached answers from `results-*.jsonl` and never calls Reactor, unless started with `REACTOR_LIVE=1`
(plus `REACTOR_BASE_URL`, `REACTOR_AGENT_ID`, `REACTOR_ACCESS_KEY` in the environment), which adds an "Ask
Reactor" button. **Don't use live mode while a benchmark runs on the same agent**: asking starts a new Reactor
history, which ends the benchmark's.

## 12. Limitations and next steps

- **Latency.** Reactor takes 60–100 s per question (median by position in a history); we also saw a 250 s answer
  late in a history and reported the pattern to the organisers. A test run takes hours.
- **Abstention calibration.** We abstain only when Reactor does. The top retrieval score barely separates
  unanswerable questions and shifts with history size; the missing-words hint (v2) is a better signal.
- **Dates in the text.** "Last month" or "three weeks ago" are passed through (v2) but not converted to dates in
  code; that would make ordering questions deterministic.
- **A local reranker** (cross-encoder or embeddings) on the BM25 candidates, for questions worded differently
  from the conversation (siblings vs brother, hikes vs trail), where dev-large retrieval drops most.
- **Running v2 on test**, once there is Reactor time.
