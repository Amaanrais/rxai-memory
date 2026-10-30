# Release format

Each split (`dev`, `dev-large`, `test`) is a directory of JSON lines files. Dev splits include `gold.jsonl`; test does not.

## Public

`histories.jsonl`, one line per user history:

```json
{"history_id": "h_...", "sessions": [
  {"session_id": "s_...", "date": "2023/05/20 (Sat) 02:21", "turns": [
    {"turn_id": "t_...", "role": "user", "content": "..."},
    {"turn_id": "t_...", "role": "assistant", "content": "..."}
  ]}
]}
```

Sessions are in chronological order. Dates keep the LongMemEval string format. A session id names the same content in every history it appears in, while its date belongs to the history: the same session can carry different dates in different histories. No session is dated after its question.

`questions.jsonl`, one line per question:

```json
{"question_id": "q_...", "history_id": "h_...", "question": "...", "question_date": "2023/05/30 (Tue) 08:08"}
```

## Gold (dev and dev-large only)

`gold.jsonl`:

```json
{"question_id": "q_...", "question_type": "knowledge-update", "abstain": false,
 "answer": "25:50", "aliases": ["25 minutes and 50 seconds"], "evidence": ["s_...", "s_..."]}
```

- `abstain` is true when the history does not contain the answer; `answer` then holds the explanation and `evidence` is empty.
- `evidence` lists session ids. Turn-level evidence is not scored: LongMemEval's turn flags miss about 10% of evidence sessions.

## Submission

`predictions.jsonl`, one line per test question:

```json
{"question_id": "q_...", "abstain": false, "answer": "...", "citations": ["s_..."]}
```

- `question_id` (string): from `questions.jsonl`. Unknown or duplicate ids make the whole file invalid; a missing question scores as wrong.
- `abstain` (JSON `true`/`false`): true when your system decides the history does not contain the answer. Abstaining on an answerable question, or answering an unanswerable one, counts as wrong.
- `answer` (string, required even when abstaining; `""` is fine then): free text from Reactor.
- `citations` (list of strings): session ids (`s_...`) from the question's history, scored as evidence F1 against the gold sessions on every answerable question.

`score.py` checks this format and scores a dev submission.

## Mapping from LongMemEval (dev splits)

| LongMemEval | Release |
| --- | --- |
| one record per question, haystack embedded | histories and questions linked by `history_id` |
| `haystack_session_ids`, `haystack_dates`, `haystack_sessions` (parallel lists) | `sessions: [{session_id, date, turns}]` |
| turn `{role, content, has_answer}` | turn `{turn_id, role, content}`; `has_answer` dropped |
| `answer_session_ids` | gold `evidence` |
| `_abs` suffix on `question_id` | gold `abstain` |
| `question_type` | gold `question_type` |
| readable ids | opaque ids |
