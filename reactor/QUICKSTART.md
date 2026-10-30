# Quickstart: your team's Reactor agent

This folder is the participant kit (it also comes as a ZIP, `reactor-hackathon-kit`): everything you need besides
one Python package.

| File | What it is |
|---|---|
| `client.py` | asks your agent one question with your evidence; prints one JSON line |
| `predictions.py` | turns the client's lines into the `predictions.jsonl` you submit |
| `PARTICIPANTS.md` | the full guide: histories, evidence, answers, errors |
| `examples/` | made-up example files, and `run_benchmark.py`, a minimal driver to start from |

From the organizers you need: your **base URL** (`https://tNN.api.reactor-hackathon.rxai.dev`, `tNN` being your
team's own slot), your **agent id** (`agt_...`) and your **access key** (`rxk_...`, sent to you privately: never commit it,
paste it in a chat or put it on a command line).

## Linux / macOS

```sh
cd reactor                                          # this folder (from the ZIP: cd reactor-hackathon-kit)
python3 --version                                   # 3.11 or newer
python3 -m venv .venv && . .venv/bin/activate
pip install "websockets>=14"

python client.py --help                             # works offline
python predictions.py examples/results-example.jsonl --histories examples/histories-example.jsonl \
    --out predictions-example.jsonl --abstain-on agent,no-answer
#   -> 3 prediction(s) (2 abstained) written to predictions-example.jsonl; 0 result(s) skipped

export REACTOR_ACCESS_KEY=rxk_...                   # from your private message
B="--base-url https://tNN.api.reactor-hackathon.rxai.dev --agent-id agt_..."
python client.py $B --new-history                   # -> {"status": "history", "history_id": "9f2c...", ...}
python client.py $B --history 9f2c... --question "When is the wedding?" \
    --question-date "2023/05/30 (Tue) 08:08" --evidence examples/excerpts-example.json \
    --question-id q_example0001 >> results.jsonl
```

## Windows (PowerShell)

```powershell
cd reactor                                          # this folder (from the ZIP: cd reactor-hackathon-kit)
python --version                                    # 3.11 or newer (if "python" isn't found: py --version)
python -m venv .venv
.venv\Scripts\python.exe -m pip install "websockets>=14"

.venv\Scripts\python.exe client.py --help           # works offline
.venv\Scripts\python.exe predictions.py examples\results-example.jsonl --histories examples\histories-example.jsonl --out predictions-example.jsonl --abstain-on agent,no-answer

$env:REACTOR_ACCESS_KEY = "rxk_..."                 # from your private message; this window only
$B = @("--base-url", "https://tNN.api.reactor-hackathon.rxai.dev", "--agent-id", "agt_...")
.venv\Scripts\python.exe client.py @B --new-history
.venv\Scripts\python.exe client.py @B --history 9f2c... --question "When is the wedding?" --question-date "2023/05/30 (Tue) 08:08" --evidence examples\excerpts-example.json --question-id q_example0001 >> results.jsonl
```

- `$B` must be a list, `@(...)`, passed as `@B`: a single string would reach the client as one argument.
- No need to activate the venv (`Activate.ps1` is often blocked by the execution policy): call
  `.venv\Scripts\python.exe` directly, as above.
- `>> results.jsonl` in Windows PowerShell 5.1 writes UTF-16. That's fine: `predictions.py` and `client.py` read
  UTF-8 and UTF-16 files.

## The whole benchmark

**One Reactor history per dataset history.** For each `history_id` (`h_...`) in `questions.jsonl`:
1. start a new Reactor history (`--new-history`);
2. ask all of that dataset history's questions in it, one at a time (`--history <Reactor history_id>`).

Never start a new history between questions of the same dataset history. Evidence: short excerpts, each with its
`session_id` as `"id"` and its session `"date"`, assistant turns included (`PARTICIPANTS.md`, section 3). You may
also load a whole dataset history into its Reactor history before asking: allowed, and described there too.

`examples/run_benchmark.py` does exactly this, appending each answer to `results.jsonl` as it goes (run it again
after an interruption: answered questions are skipped). Its `retrieve()` is a naive keyword baseline: **replace it
with your retrieval.**

```sh
python examples/run_benchmark.py --questions questions.jsonl --histories histories.jsonl \
    --base-url https://tNN.api.reactor-hackathon.rxai.dev --agent-id agt_... --out results.jsonl
python predictions.py results.jsonl --histories histories.jsonl --out predictions.jsonl
```

(PowerShell: the same two commands on one line each, with `.venv\Scripts\python.exe` and `examples\run_benchmark.py`.)

`predictions.py` exits 1 when it skipped questions (agent abstentions and unanswered ones are skipped unless you
choose `--abstain-on agent` and/or `no-answer`); the list is on screen. See `PARTICIPANTS.md`, section 6.

## When something fails

Send the support channel the JSON line the client printed: it never contains your key. `PARTICIPANTS.md` ends with
a troubleshooting table.
