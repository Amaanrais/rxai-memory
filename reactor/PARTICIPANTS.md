# Asking your team's Reactor agent

Your team gets from the organizers (through the event platform):
- its **base URL**, `https://tNN.api.reactor-hackathon.rxai.dev`, where `tNN` is your team's own slot (`t07`, say).
  Every team has its own host and its own agent; a key works only on its own team's host.
- its **agent id(s)** (`agt_...`): the agent behind that host.
- its **access key** (`rxk_...`), sent to your team privately. Keep it secret: never commit it, paste it into a
  chat or put it on a command line (step 2). The URL and agent id are not secret; the key is.
- the **support channel**. Questions or trouble: ask there, with the JSON line the client printed (it never contains
  your key).

`client.py`, `predictions.py` and the examples are in this folder (the same kit also comes as a ZIP).

`QUICKSTART.md` has the same steps as copy-paste commands for Linux/macOS and for Windows PowerShell. The examples
below use Linux/macOS shell syntax.

## 1. Install

Python 3.11 or newer, and one package:

```sh
python -m venv .venv
# macOS/Linux:          source .venv/bin/activate
# Windows PowerShell:   .venv\Scripts\Activate.ps1
pip install "websockets>=14"
```

## 2. Set your key

Put it in the environment, never on the command line:

```sh
export REACTOR_ACCESS_KEY=rxk_...           # macOS/Linux
$env:REACTOR_ACCESS_KEY = "rxk_..."         # Windows PowerShell
```

## 3. The workflow: one Reactor history per dataset history

Two different things are called a "history", with two different ids:

| | Looks like | Where it comes from | What it is |
|---|---|---|---|
| **Dataset history** | `"history_id": "h_..."` | `questions.jsonl`, `histories.jsonl` | one user's past chat sessions, which questions are about |
| **Reactor history** | `"history_id": "9f2c..."` | `client.py --new-history` | your agent's memory; you pass it as `--history` |

The agent remembers. Within a Reactor history it keeps everything it has seen; between two it keeps nothing. The
rule:
- **One Reactor history per dataset history.** Group `questions.jsonl` by `history_id`. For each dataset history,
  run `--new-history` once, then ask **all** of that dataset history's questions in that Reactor history, one at a
  time. Several questions can share one dataset history.
- **Reset only between dataset histories.** Run `--new-history` again only when you move on to the next dataset
  history. Never reset between two questions of the same dataset history, and never ask a question in a Reactor
  history you started for another dataset history: its memory holds another user's sessions.

Two ways to give the agent what it needs, and you can combine them:
- **Evidence with each question** (below): your retrieval finds the excerpts, and you send them with the question.
- **Load the whole dataset history** into the Reactor history first, then ask (*Loading a whole dataset history*,
  below). Allowed: in its simplest form, it's the baseline to beat.

```sh
B="--base-url https://tNN.<domain> --agent-id agt_..."
# Windows PowerShell: $B = @("--base-url", "https://tNN.<domain>", "--agent-id", "agt_..."), then client.py @B ...

# a) the next dataset history (h_1a2b3c4d...): start a new Reactor history -- memory, chat history, pending input
#    and late replies are wiped
python client.py $B --new-history
#   {"status": "history", "history_id": "9f2c...", "running": true, ...}

# b) each question of h_1a2b3c4d..., in that Reactor history, with its question_date exactly as in questions.jsonl,
#    your evidence for it, and its question_id; append each line to your results file
python client.py $B --history 9f2c... --question "When was my sister's wedding?" \
    --question-date "2023/05/30 (Tue) 08:08" --evidence q1.json --question-id q_3f9a... >> results.jsonl
#   {"status": "answered", "question_id": "q_3f9a...", "answer": "In June [E2].",
#    "citations": [{"label": "E2", "id": "s_81c2..."}], ...}
python client.py $B --history 9f2c... --question "Where was it held?" \
    --question-date "2023/05/30 (Tue) 08:08" --evidence q2.json --question-id q_7b10... >> results.jsonl

# c) the next dataset history: back to a)
```

**Evidence.** `q1.json` is a UTF-8 file (with or without a BOM): a list of excerpts, each with its session's id and
date.

```json
[{"id": "s_07de...", "date": "2023/05/20 (Sat) 02:21",
  "text": "user: We booked the hall for May.\nassistant: Great, May is a good month for it."},
 {"id": "s_81c2...", "date": "2023/05/27 (Sat) 10:05",
  "text": "user: The wedding moved to June, the hall is the same."}]
```

- **Send excerpts, not whole sessions.** A session is often around 10,000 characters and can pass 20,000, so one
  whole session can exceed the message limit (below). Send the turns that carry the answer, with a little context
  around them.
- **`id` is the excerpt's `session_id`** from `histories.jsonl`, even when you retrieved it by turn: the evaluator's
  citations are session ids. A `turn_id` (`t_...`) is not a citation (`predictions.py` leaves it out). Several
  excerpts from one session share its id.
- **`date` is that session's `date`**, copied as written from the dataset history you're answering. The same session
  can appear in several dataset histories with different dates; take the one from this history. The client passes
  it to the agent next to the excerpt. Questions about order, "how long ago" or "the most recent" need these dates.
- **Keep assistant turns.** Some questions ask what the assistant said or recommended, so their answer is only in
  an assistant turn. Retrieve from both roles, and mark who's speaking (`user:` / `assistant:`), as above.

A list of plain strings works too, and so does a plain-text file with passages separated by blank lines, but their
citations then have `"id": null` and can't be submitted. `--evidence -` reads stdin; on Windows, pass a file instead.

**Question date.** Pass `--question-date` the `question_date` from `questions.jsonl` as it is,
`"2023/05/30 (Tue) 08:08"` (quoted: it has spaces). `2023-05-30` and `2023-05-30 08:08` work too. The weekday is
checked against the date. The time is kept, and no timezone is added: the dataset gives none. The agent is told
to treat that moment as now. Its own clock shows the real date, so the client tells it to ignore that clock for
this question. The output gives it as `"question_date": "2023-05-30 08:08"`.

**Message limits.** Question, date, excerpts and the client's instructions travel as one message:
- The client refuses one over **16,000 characters** before sending anything (exit 2). `--max-chars` raises that.
- The server refuses one over **its limit**, 32,000 characters when the event starts (support announces any
  change), whatever the flag (exit 4, `too_large`, with `limit` and `size`). Nothing reaches the agent.

Longer messages cost the agent more time on every step; send the excerpts you'd stake the answer on.

**Loading a whole dataset history.** After `--new-history`, send the dataset history's sessions with `--text`
(sent as written: no template, no final answer expected), oldest first, a part at a time, then ask its questions
with `--question` as above:

```sh
python client.py $B --history 9f2c... --text "Part 1 of 125 of my past chats, to remember. Reply only OK.
[session s_07de..., 2023/05/20 (Sat) 02:21]
user: We booked the hall for May.
assistant: Great, May is a good month for it.
..."
#   {"status": "reply", "reply": "OK", ...}
```

- **Each part needs a reply before the next one is accepted.** The client waits for the agent's first reply to
  your part (`"status": "reply"`, exit 0). While a part has no reply, the next is refused (`previous_pending`), and
  the client keeps retrying within `--timeout`. So end each part asking for a short acknowledgement, as above.
- **Label each session with its `session_id` and `date`** in the text: temporal questions need the dates, and the
  agent can then name the sessions it used.
- **It takes many parts.** A test history is about 200 sessions, around 2 million characters: about 125 parts at the
  client's 16,000 characters, or 65 at the server's 32,000 (`--max-chars 32000`). On Windows a whole command line is
  limited to about 32,000 characters, so keep parts there under about 30,000. Every part is a full turn of the
  agent: loading a history takes time, so start early.
- **What's loaded stays in that Reactor history only.** `--new-history` wipes it, as it does everything else.
- **Citations still come only from `--evidence`.** The client maps `[E#]` labels to your ids; an answer asked
  without `--evidence` has no citations, and evidence F1 counts on every answerable question. After loading, still
  send the excerpts your retrieval found, so that the answer can cite them.
- **Keep loading lines out of `results.jsonl`**: only `--question` lines with a `--question-id` become predictions.

**Browsing.** Your agent can browse public web pages. If you run your own search over a public URL, tell the agent
about it in your question or in a loaded part, and it can query it. It reaches only what's public: never put your
access key, or anything you'd mind others seeing, into such a URL. With `--evidence`, the client asks the agent to
answer from that evidence; to have it search as well, say so in the question.

## 4. How the answer is recognized

The client asks the agent to write its answer on a line starting with `# FINAL ANSWER [ref:<8 hex>]:`, with a fresh
ref for every question. **Only that line counts as the answer.**
- The agent may first send interim notes ("let me look"). They go to `other_replies`, even if they mention the ref.
- A final answer to an *earlier* question that arrives late goes to `late_replies` with its ref. It is never taken
  as this question's answer.
- If the information it has doesn't answer the question, the agent is asked to write
  `# FINAL ANSWER [ref:...]: INSUFFICIENT_EVIDENCE`. You get `"abstained": true`.
- If the agent never writes the marked line, you get `"status": "timeout"`, with whatever it did say in
  `other_replies`.

## 5. Results

One JSON line on stdout. It's ASCII: other characters come as `\uXXXX`, which any JSON parser turns back into text.

| Exit | `status` | Meaning |
|---|---|---|
| 0 | `answered` | `answer`, `abstained`, `citations` (labels the answer cites, with your ids), `unknown_citations` (labels you never sent), `uncited` (evidence sent, none cited), `reply`, `cycle`, `latency_s`, `retries` |
| 0 | `history` | `--new-history` worked: `history_id`, `running` |
| 2 | `error` | refused at connect (wrong key or agent, agent not running), a new history refused, bad input, or Python/websockets too old: `error` says which |
| 3 | `timeout` | no final answer within `--timeout` seconds |
| 4 | `refused` | the server refused this question: `code` (table below) |
| 5 | `closed` | the chat closed before an answer: a new history started, the agent stopped, access was revoked, or the agent restarted after its model provider failed. Ask again; if it repeats, tell support |

Every line also carries `other_replies` and `late_replies` where they apply, plus `ref`, `question`,
`question_date`, `question_id` and `history_id` (the Reactor history you named) when given. From your own code: `subprocess.run([...],
capture_output=True, text=True)`, then `json.loads(proc.stdout)`, whatever the exit code.

`--timeout` (or `REACTOR_TIMEOUT`) sets the wait in seconds, retries included. The default is 300; support
announces a recommended value once the organizers have measured real answer times.

## 6. Your submission: `predictions.py`

The evaluator takes `predictions.jsonl`, one line per question:

```json
{"question_id": "q_3f9a...", "abstain": false, "answer": "Hawaii", "citations": ["s_81c2...", "s_07de..."]}
```

```sh
python predictions.py results.jsonl --histories histories.jsonl --out predictions.jsonl
```

It maps each client line and invents nothing. Everything it leaves out is listed, and exit 1 says something was.
- `question_id`: the one you passed with `--question-id`. A line without one is skipped.
- `answer`: the final answer without its `[E#]` labels (`In June [E2].` becomes `In June.`).
- `citations`: the session ids of the passages the answer cites, in order, without repeats. A cited passage with
  no id, an id that isn't a session id in `histories.jsonl`, and a label you never sent are left out.
- **Abstaining is your decision.** By default, a question the agent answered `INSUFFICIENT_EVIDENCE` and a
  question with no answer (timeout, closed, refused, error) are **skipped**, not abstained. Add
  `--abstain-on agent` to submit the agent's abstentions as `"abstain": true` (empty answer, no citations),
  `--abstain-on no-answer` for unanswered questions, or both: `--abstain-on agent,no-answer`.
- Two lines for one question stop it, and nothing is written: keep the one you mean (e.g. after a retry).

`--histories` collects every `session_id` field in `histories.jsonl`; if the file names it differently,
`--session-key`.

## 7. One question at a time, and parallelism

The agent is stateful, so the server never merges two questions. It refuses instead (exit 4, `code`), and every
refusal says the question never reached the agent (`"forwarded": false`):

| `code` | Why | What to do |
|---|---|---|
| `history_required` | no history yet | `--new-history` |
| `history_changed` | a newer history has started since | the agent's current one is `current_history_id`; ask there, or start another |
| `history_resetting` | a new history is starting right now | retry in a moment |
| `busy` | another chat with this agent is open | the client retries by itself (below) |
| `previous_pending` | the agent hasn't yet taken up or replied to your previous question | the client retries by itself; if it persists, `--new-history` |
| `too_large` | the question is over the server's `limit` | send fewer or shorter passages |

`busy` and `previous_pending` are normal for a few seconds right after an answer: the agent is finishing its thought.
The client asks again by itself, backing off 1, 2, 4 and 8 seconds within `--timeout`, and reports `retries`.

**Parallelism exists only across agents.** If your team was given several agent ids, run one history at a time on
each, in parallel. On a single agent, ask one question at a time.

## Troubleshooting

| Message | What to do |
|---|---|
| `set REACTOR_ACCESS_KEY` | step 2, in the same terminal |
| `refused at connect (HTTP 403)` | check the key and agent id; otherwise the agent isn't running: ask support |
| `refused at connect (HTTP 404)` | `--base-url` is just `https://HOST`, without a path |
| `getaddrinfo failed` / `Name or service not known` / `nodename nor servname` | the host in `--base-url` doesn't resolve: check it letter by letter against your handout; if it's right, ask support (the API may not be open yet) |
| `new history refused (HTTP 409)` | a new history or maintenance is already under way: retry in a moment |
| `new history refused (HTTP 503): still_stopping` | the agent's last run is still shutting down: nothing was changed; retry in a minute |
| `new history refused (HTTP 503): restart_failed` | the new history is in place but the agent didn't start: retry, or ask the organizers |
| `use https://` | the base URL must start with `https://` |
| `predictions.py`: `no 'session_id' field found` | `--histories` names the evaluator's `histories.jsonl`; if its field is named otherwise, `--session-key` |
| `evidence: not UTF-8 text` | save the file as UTF-8 |
| `--question-date ...: use YYYY-MM-DD, ...` / `is a Tue, not ...` | pass `question_date` from `questions.jsonl` as it is, in quotes |
| `the message is N characters, over --max-chars` | send shorter excerpts: the turns that matter, not whole sessions |
| `needs Python 3.11 or newer` / `needs ... websockets` | step 1 |
