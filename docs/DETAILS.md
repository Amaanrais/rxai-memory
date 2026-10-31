# Recall: details

The full results, design and history behind the [README](../README.md). The organisers' brief is
[CHALLENGE.md](../CHALLENGE.md); the data format is [format.md](../format.md).

1. [Results](#1-results)
2. [Pipeline and design decisions](#2-pipeline-and-design-decisions)
3. [Versions: v1, v2, v3](#3-versions-v1-v2-v3)
4. [Driver behaviour](#4-driver-behaviour)
5. [The test runs](#5-the-test-runs)
6. [Reproducibility](#6-reproducibility)
7. [Offline evaluation](#7-offline-evaluation)
8. [Demo app](#8-demo-app)
9. [Limitations and next steps](#9-limitations-and-next-steps)

## 1. Results

### Test run

All 146 test questions were answered by Reactor (`results-test-v3.jsonl`, 147 result lines: one timeout, asked
again).

| | first run (v1, `predictions-v1.jsonl`) | submission (v3, `predictions.jsonl`) |
|---|---|---|
| answers, each citing its sources | 115 | **123** |
| abstained | 31 of 146 (21%) | **23 of 146 (16%)** |
| citations from | | Reactor's own labels on the answer line, for all 123 |
| Reactor histories | | 16 for 15 dataset histories (one split, [section 5](#5-the-test-runs)) |

The test split has unanswerable questions too, so some of the 23 abstentions are correct; the official scores will
tell how many.

### Retrieval on dev

Offline, `eval_retrieval.py`, saved in `metrics/`. *hit@5*: a gold session is in the top 5; *sent*: share of gold
sessions in the evidence sent; *ans∈ev*: the gold answer (or an alias, normalized as `score.py` does) appears in
the evidence sent. v3 (`metrics/retrieval-dev-v3.json`), answerable questions:

| dev, by type | n | hit@5 | sent | ans∈ev |
|---|---|---|---|---|
| knowledge-update | 72 | 1.000 | 0.993 | 0.819 |
| multi-session | 121 | 0.983 | 0.913 | 0.446 |
| single-session-assistant | 112 | 1.000 | 1.000 | 0.589 |
| single-session-user | 64 | 0.984 | 0.984 | 0.875 |
| temporal-reasoning | 127 | 0.953 | 0.916 | 0.323 |
| **all** | 496 | **0.982** | **0.954** | **0.556** |

dev has ~50 sessions per history. On dev-large (~500 sessions per history; the first 130 questions, 124
answerable; `metrics/retrieval-dev-large-130-v3.json`): hit@5 0.935, sent 0.878, ans∈ev 0.492.

| | hit@5 | sent | ans∈ev v1 | ans∈ev v2 | ans∈ev v3 |
|---|---|---|---|---|---|
| dev (496 answerable) | 0.982 | 0.954 | 0.542 | 0.556 | 0.556 |
| dev-large (124 answerable) | 0.935 | 0.878 | 0.484 | 0.492 | 0.492 |

hit@5 and sent are the same in all three versions. v3 changes only name questions; on the 12 answerable name
questions in dev-large, ans∈ev rises from 0.833 to 0.917 (hit@5 unchanged at 0.917); on dev's 12, nothing changes.

### v1 vs v3 with Reactor on 24 hard dev-large questions

`sample_hard.jsonl`: 10 name questions, 4 temporal, 2 knowledge-update, 2 multi-session counts and 6 unanswerable.
v1 is `run.py` (`results-hard-v1.jsonl`); v3 is `run_v3_test.py` with the submission's settings
(`results-hard-v3.jsonl`). Scored by `compare.py` with `score.py`'s matcher:

| | v1 | v3 |
|---|---|---|
| correct | 15 of 24 | **16 of 24** |
| unjudged (left to the official LLM judge) | 4 | 4 |
| false abstentions (of 18 answerable) | 5 | **4** |
| unanswerable questions abstained on | 6 of 6 | 6 of 6 |
| evidence F1 | 0.67 | **0.75** |

By kind: name questions correct 6 → 7 of 10 (false abstentions 2 → 1, evidence F1 0.80 → 0.90); knowledge-update
evidence F1 0.67 → 0.83; temporal and multi-session unchanged.

### Reactor on a first dev sample

`sample10` (2 questions per type plus 1 unanswerable), v1, scored with `score.py`. It was run before v2 and
motivated it.

| type | n | correct | unjudged | abstention recall | false abstention | evidence F1 |
|---|---|---|---|---|---|---|
| knowledge-update | 2 | 0.500 | 0.500 | – | 0.000 | 0.667 |
| multi-session | 3 | 0.333 | 0.333 | 0.000 | 0.000 | 0.733 |
| single-session-assistant | 2 | 0.000 | 0.500 | – | 0.500 | 0.333 |
| single-session-user | 2 | 1.000 | 0.000 | – | 0.000 | 1.000 |
| temporal-reasoning | 2 | 0.000 | 1.000 | – | 0.000 | 0.667 |
| **all** | 11 | **0.364** | 0.455 | 0.000 | 0.100 | **0.680** |

Its failures, diagnosed by hand: an answer cut off by the excerpt budget, a near-miss trap answered instead of
abstaining, a two-session difference, and an event order needing a date from elsewhere in a session.

### Efficiency

`efficiency.py` counts the whole message per question (question, instructions, excerpts and the client's template,
built with `reactor/client.py`'s `build_message`) from each run's evidence log, against a full test history of
1.66M characters (median; 1.63M to 1.73M):

| | v1 | v3 (submitted) |
|---|---|---|
| message per question, median | 8.6k characters | 11.3k characters |
| message per question, max | 20.7k | 21.0k |
| per question, less than the full history (median) | 190× | **147×** |
| per dataset history, all its questions together (median) | 85k | 114k |
| per dataset history, less than loading it once (median) | 20× | **15×** |
| per dataset history, range | 12–26× | 11–25× |

v3 sends more because the 48 name questions get 8 sessions: their median evidence goes from 5.7k to 11.2k
characters, while the other 98 questions stay at about 8.7k. Per dataset history, each question is counted once (a
retry sends the same message). Loading a history also takes 65–125 messages (at 32k or 16k characters each), each a
full agent turn. The v3 column comes from the committed `results-test-v3.evidence.jsonl`; v1's evidence log,
`results-test.evidence.jsonl`, is not in the repository.

## 2. Pipeline and design decisions

- **Turn-level BM25 with session aggregation.** Each turn of both roles is a chunk, prefixed `user:` or
  `assistant:`; long turns are split into 1,500-character windows. A session scores its best chunk plus 0.25 × its
  next two. Turns, not sessions, because a session is ~10k characters and the fact is usually in one turn.
- **Assistant turns are indexed.** "What did you recommend…" questions are answered only by assistant turns.
- **Question routing.** Counting, totalling, comparing and ordering questions ("how many", "which first", "how
  long ago"), and in v3 name questions, get 8 sessions in 24,000 characters (`--max-chars 30000`); others 5
  sessions in 12,000.
- **Dates done in code.** Each excerpt carries its session date and "(N days before the question date)"; the latest
  excerpt is marked `[most recent]`, so knowledge-update questions take the latest value.
- **Answer instructions.** Every question: put the `[E#]` labels on the FINAL ANSWER line, one value, no hedging.
  Questions about a value that may have changed: "the most recent date wins", except when the question asks for
  the previous or original value.
- **Citation rules**, per the organisers' ruling that fallback citations must support Reactor's answer
  (`make_predictions.py`):
  1. Reactor's `[E#]` labels on the answer line;
  2. else the labels in the last paragraph of its reasoning;
  3. else a support check: the excerpts that contain Reactor's answer (normalized as `score.py` does), or ≥ 60% of
     its content words, and only if that picks out at most half of the excerpts (a bare "2" matches all of them, so
     it cites nothing);
  4. abstentions cite nothing. We never fall back to the top-ranked session.

  Labels map to session ids through the run's evidence log (E1 is the first excerpt), or the client's own mapping
  when there is no log. A question abstains when Reactor answered INSUFFICIENT_EVIDENCE, its status isn't
  "answered", or it has no result line.

## 3. Versions: v1, v2, v3

**v1** (`retrieval.py` `version=1`, `run.py`) is the pipeline in section 2.

**v2** (`version=2`, `run_v2.py`) adds:

- **Focused trimming**: a long turn keeps the stretch with the most question words, not its beginning.
- **Lead-session budget**: a clearly leading session gets up to half the budget.
- **Time sentences**: for temporal and counting questions, the user's sentences that say when something happened
  ("last month", "three weeks ago"), from anywhere in a sent session.
- **Missing-words hint** for near-miss traps: "the evidence never mentions: X. If the answer depends on these,
  answer INSUFFICIENT_EVIDENCE".
- **Work-it-out prompt** for aggregate questions: compute a difference, total, count or order before the answer
  line.

The missing-words hint fires on 15 of 30 (50%) unanswerable dev questions, against 62 of 496 (12.5%) answerable
ones; on dev-large, 3 of 6 against 2 of 124 (1.6%). A tried change, dropping a session's weaker second hit,
lowered ans∈ev and was left out.

**v3** (`version=3`, `run_v3.py`) is v2 plus, for questions asking for a name ("what is my dog's name", "who is my
instructor"):

- **Synonyms** for the entity word, at weight 0.6 (dog → puppy, pup, pet; partner → husband, wife, boyfriend,
  girlfriend; instructor → teacher, coach; …).
- **A name boost** (×1.5) for turns with a capitalised, name-like word within 40 characters of the entity word.
- **Routing to 8 sessions** in 24,000 characters.

Other questions get exactly v2's evidence.

**Why v3 with these settings.** v1's test run abstained on 31 of 146 questions (21%), all of them Reactor answering
INSUFFICIENT_EVIDENCE; 48 test questions ask for a name, and 13 of those were abstentions. So the submission runs
v3 with `run_v3_test.py`'s defaults: **no missing-words hint** (`--hint` puts it back), and a **lenient abstain
line** (`--strict` leaves it out): "Answer INSUFFICIENT_EVIDENCE only if nothing in the evidence addresses the
question. If the evidence gives a likely answer, give it." The other instructions stay as in v2.

## 4. Driver behaviour

Following the organisers' guidance:

- **One Reactor history per dataset history**, started with `--new-history`, never reset mid-history.
- **`--timeout 600`** for every question.
- **A timeout doesn't cancel the question on Reactor's side**, so the driver first sends nothing and listens on the
  same Reactor history for up to 120 s for its late answer (`listen.py`, with the client's backoff). Only if none
  arrives is the question asked once more in the same history; the client keeps retrying while the previous
  request hasn't cleared (`previous_pending`).
- **Late answers are recovered.** If Reactor's final answer to a timed-out question arrives later, in another
  question's `late_replies` (matched by its ref), it is recorded as that question's answer, marked `recovered`.
  `make_predictions.py` keeps exactly one line per question: the original question's late answer over a retry's,
  otherwise the earlier of two answered lines.
- **Wait for the agent after a new history.** A new history restarts the agent, and a question sent before it is
  back up fails at once (chat closed, HTTP 403 or `history_resetting`). So the driver waits 15 s (`--settle`) after
  `--new-history`, and a question that fails that way before anything is answered in its Reactor history is asked
  again in the same history after 10, 20 and 40 s. Every attempt is recorded. In the v3 test run no question
  needed it.
- **Resume without a reset.** A rerun asks only the questions not yet answered. The dataset history in progress
  continues in its Reactor history (the agent's current one), before any other history is started.

## 5. The test runs

**v3, the submission** (`run_v3_test.py`, code at commit `5ada236`; `retrieval.py`, `run.py`, `run_v3_test.py` and
`make_predictions.py` are unchanged since apart from docstrings). Reactor's answers took 2.7 h in total (median
63 s per question, max 140 s); with the timeout and the waits after each new history, the run took about 4 h.

One dataset history (`h_8224cd1fdcd0`, 10 questions) was split across two Reactor histories. Its fifth question,
`q_3664132a79f0`, timed out after 600 s; no late answer arrived while the driver listened, and when it was asked
again in the same Reactor history the server reported `history_changed`. The driver started a new Reactor history
for that dataset history and asked it and the remaining five questions there. All 10 were answered. The other 14
dataset histories each used one Reactor history, and the run had no other timeout, error or refusal.

**v1, the first run** (`run.py`; `results-test.jsonl`, `predictions-v1.jsonl`):

| Result lines | Driver | Notes |
|---|---|---|
| 1–32 | tag `v1-test-run` | client timeout 300 s (the kit's default); a timeout was retried after 30 s without listening |
| 33 onwards | tag `v1.1-test-run` | after a mid-run restart: driver changes only |

Between the two tags, v1's evidence and question text are byte-identical (all 526 dev questions through a
recording stand-in for the client: 0 differences; identical evidence logs, 9,050,462 bytes). Two questions
(`q_1a6b4fa2ede3`, `q_aa27ddd3b7c5`) timed out under `v1-test-run` and no late answer arrived. Their Reactor
history had already been replaced at the restart, so they were asked again in a fresh Reactor history for their
dataset history, the kit's documented rerun behaviour (`reactor/QUICKSTART.md`). Every question carries its own
evidence, so nothing from the earlier history is needed.

## 6. Reproducibility

- From the committed `results-test-v3.jsonl`, `make_predictions.py` rebuilds `predictions.jsonl` byte for byte,
  without calling Reactor (the evidence log isn't needed: every answer cites by Reactor's own labels).
- `tests/test_run_v3_test.py` proves, with a fake client (`tests/fake_client.py`), that `run_v3_test.py` sends
  exactly what `run_v3.py --no-hint --lenient` sends: question text, date, `--max-chars` and evidence bytes. The
  unit test covers the first 40 dev questions; `python tests/test_run_v3_test.py <questions> <histories>` runs any
  set in full (checked on all 526 dev questions and on `sample_hard.jsonl`).
- The final clean-up changed docstrings and comments only: the code is identical once docstrings are removed, and
  `run.py` and `run_v3_test.py` send byte-identical client calls and evidence logs on dev and `sample_hard.jsonl`
  before and after.

## 7. Offline evaluation

No Reactor calls:

```sh
python eval_retrieval.py dev --version 3                      # retrieval metrics (add --json FILE to save)
python eval_retrieval.py dev-large --limit 130 --version 3
python compare.py results-hard-v1.jsonl results-hard-v3.jsonl --questions sample_hard.jsonl --gold dev-large/gold.jsonl
python efficiency.py results-test-v3.evidence.jsonl --questions test/questions.jsonl --histories test/histories.jsonl
python -m unittest discover tests
```

`score.py dev/gold.jsonl predictions.jsonl` scores a full dev run (`run_v3_test.py` and `make_predictions.py` with
`dev/` in place of `test/`; that calls Reactor). It scores against the whole gold file, so for a subset such as
`sample_hard.jsonl` use `compare.py`.

## 8. Demo app

```sh
streamlit run app.py --server.address localhost
```

Recall's chat interface: ask about a user's history and see the answer, the conversations it came from (cited
passages highlighted) and how it was answered (what was sent, and how many characters against the whole history).
An About page covers who it's for, how it works and the results. Saved answers come from the `results-*.jsonl`
files (the start screen's suggestions from `results-sample10.jsonl` and `results-demo4-v3.jsonl`); retrieval runs
live for any question. By default the app never calls Reactor.

Live mode (`REACTOR_LIVE=1`, with `REACTOR_BASE_URL`, `REACTOR_AGENT_ID` and `REACTOR_ACCESS_KEY` in the
environment) asks Reactor directly. Don't use it while a benchmark runs on the same agent: asking starts a new
Reactor history, which ends the benchmark's.

## 9. Limitations and next steps

- **Latency.** Reactor takes about a minute per question, so a test run takes hours on one agent.
- **Abstention vs. traps.** The lenient prompt lowered abstentions from 21% to 16%; without test gold we can't yet
  say whether it lets near-miss traps through. The missing-words hint is the guard to bring back if it does.
- **Wording gaps.** Questions worded differently from the conversation (siblings vs. brother, hikes vs. trail) are
  where dev-large retrieval drops most; a local reranker (cross-encoder or embeddings) on the BM25 candidates would
  help.
- **Relative dates.** "Last month" or "three weeks ago" are passed through, not converted to dates in code; doing so
  would make ordering questions deterministic.
