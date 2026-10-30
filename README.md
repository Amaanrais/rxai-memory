# Memory Retrieval Challenge

You are provided conversations between a user and an assistant, spanning from weeks to months. For each user, there is a question they ask that depends on something that was said earlier: a fact, a change of plans, a date, or something the assistant recommended.

For each user, there are dozens to hundreds of these conversation - small talk, requests, long tangents, most of them unrelated to the question. Your task is to find the needle (the part relevant to the question) in the haystack (the whole conversation history).

Implement a system that answers these questions. It should:
- find the relevant parts of a long, noisy history;
- get the answer from Reactor;
- cite the conversations the answer comes from;
- abstains when the history does not contain the answer.

## Data

Download from the [Releases](https://github.com/RxAI-dev/memory-challenge/releases) page.

| Split | Histories | Gold | Use |
| --- | --- | --- | --- |
| `dev` | 526 questions, one history per question, about 50 conversations each | included | build and test your system |
| `dev-large` | the same questions, about 500 conversations per history | included | practise at test scale |
| `test` | 146 questions in 15 histories, about 10 per history, 200 conversations per history | not released | final scores |

Each split has:
- `histories.jsonl` (dated sessions of user and assistant turns)
- `questions.jsonl` (question, its history, the date it is asked)
- for dev splits, `gold.jsonl` (answer, evidence sessions, abstention, question type)
 
The format is described in [format.md](format.md).

- Sessions are in chronological order, and no session is dated after its question. A session id refers to the same conversation wherever it appears; its date belongs to the history.
- Dev splits are built from the public [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark (MIT licence). If a download fails, `convert_longmemeval.py` rebuilds them from the original files, identical to the released ones: `python convert_longmemeval.py longmemeval_m_cleaned.json dev-large`.
- The test split is new data in the same format. It contains near-miss conversations that look relevant but do not answer the question.

## Questions

| Type | What it tests | Example |
| --- | --- | --- |
| single-session-user | a fact the user mentioned once | What degree did I graduate with? |
| single-session-assistant | something the assistant said | What ratio of tea tree oil to carrier oil should I use? |
| multi-session | combining facts from several conversations | What is the total distance of the hikes I did on two consecutive weekends? |
| knowledge-update | the latest value of something that changed | What was my personal best time in the charity 5K run? |
| temporal-reasoning | dates, durations and order of events | How many days passed between my visit to MoMA and the exhibition? |

- **Some questions cannot be answered from the history**: about 6% in dev, 8% in test. Example: "What is the name of my hamster?" when the user only ever mentioned a cat. Your system should abstain on those.
- Questions do not always say which conversation they refer to.
- The question type is given in dev gold, not for test questions.

## Reactor

Reactor must provide the final answer. Your own code does everything around it: indexing, retrieval, preprocessing, deciding what to send.

- Each team gets its own Reactor agent, an access key and a participant kit with `client.py` and a minimal `run_benchmark.py` example (keyword retrieval, history resets, result collection). The kit and keys are distributed through the event platform.
- `client.py` sends your question with your passages labelled `[E1]`, `[E2]`, ..., asks Reactor to cite them, and maps the cited labels back to your passage ids. Use session ids as passage ids and the citations go straight into your submission.
- Pass the question's date with `--question-date`, and put each session's date into its passage text: temporal questions need both.
- The client limits a message to 16,000 characters of question and evidence by default. An average conversation is about 10,000 characters, so send selected turns or snippets, not whole conversations.
- Reactor has long-term memory. You are given an opportunity to reset between histories, so that different users do not mix. You can load a history into it and then ask: that is allowed, and in its simplest form it is the baseline to beat.
- Reactor can browse public web pages, so you can expose your own search over a public URL and let it query that.
- Reactor answers asynchronously, typically within tens of seconds. Plan your test run accordingly.
- Reactor runs on a fast instruct model: date arithmetic and counting may be more reliable in your own code than left to Reactor.

## Rules

- Final answers come from Reactor, produced by your submitted code. Do not edit predictions by hand.
- Your own local models and tools are allowed: embedding models, rerankers, classic search, anything you run yourself.

## Scoring

Scores are reported per question type and overall.

- **Answer accuracy**: answers are checked against the gold answer and its accepted variants by an LLM judge. Several candidate answers, hedging or negation count as wrong. Answering an unanswerable question, or abstaining on an answerable one, counts as wrong.
- **Evidence F1**: overlap between your citations and the gold evidence sessions, on every answerable question, including ones you abstained on.
- **Abstention**: how often you abstain on unanswerable questions, and how rarely you abstain on answerable ones.

How the benchmark score combines with the final presentations is announced at kickoff.

### Self-scoring on dev

```
python score.py dev/gold.jsonl predictions.jsonl
```

`score.py` checks your file format and scores it with the same metrics. It needs only the Python standard library. Answers it cannot match automatically are marked "unjudged": official scoring sends those to the judge.

## Submission

Your submission should contain three parts:

1. `predictions.jsonl`, one line per test question (format below).
2. Your code: a link to a public repository or an archive, with a short README saying how to run it. The predictions must come from running this code; organisers may re-run it on a few questions to check.
3. A short presentation of your solution at the final session.

```json
{"question_id": "q_...", "abstain": false, "answer": "...", "citations": ["s_...", "s_..."]}
```

- `answer`: free text, produced by Reactor. Required even when abstaining (`""` is fine then).
- `abstain`: `true` when your system decides the history does not contain the answer.
- `citations`: ids of the sessions the answer is based on.

The submission deadline, how to submit, and the presentation format are announced at kickoff.

## Tips

- Long histories contain a lot of noise, and the test set adds near misses on purpose. Retrieval precision matters as much as recall, because only a small part of a history fits in a message.
- For knowledge-update questions the latest mention wins; for temporal questions the session dates matter as much as the text.
- Abstaining is a decision your system makes: weak retrieval results are a signal.
