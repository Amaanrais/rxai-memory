// Everything the deck shows lives here. Change a number or a line, reload index.html.
// Source of truth for numbers: README.md (sections 7, 9, 10) and metrics/*.json.
// Metrics: 0..1 values; null = "pending" (shown as not yet measured).
const DECK = {
  title: "Recall",
  subtitle: "Grounded long-term memory on Reactor: find the few turns that matter, cite only what supports the answer",
  event: "RxAI Memory Retrieval Challenge · October 2026",
  team: ["Amaan Rais", "Dedar Alam", "Frensi Qatipi"],
  repo: "rxai-memory",

  problem: {
    stats: [
      { value: "200", label: "conversations per user (test)" },
      { value: "~1.66M", label: "characters in one test history" },
      { value: "16k", label: "characters Reactor accepts per message" },
      { value: "8%", label: "of test questions can't be answered" },
    ],
    types: [
      ["single-session-user", "a fact the user said once"],
      ["single-session-assistant", "something the assistant said"],
      ["multi-session", "combine facts across conversations"],
      ["knowledge-update", "the latest value of something that changed"],
      ["temporal-reasoning", "dates, durations, order of events"],
    ],
  },

  // The pipeline diagram, left to right (README section 2).
  pipeline: [
    { name: "Index", detail: "every turn, both roles, BM25 per history" },
    { name: "Route", detail: "count / order / time: 8 sessions, 24k; else 5, 12k" },
    { name: "Retrieve", detail: "best turns ± 1, dates, N days before, [most recent]" },
    { name: "Reactor", detail: "answers from [E1]..[En], labels on the answer line" },
    { name: "Cite / abstain", detail: "labels, else support check; INSUFFICIENT_EVIDENCE → abstain" },
  ],

  design: [
    ["Turn-level BM25", "A session is ~10k characters but the fact is usually in one turn. Sessions score their best turn + 0.25 × the next two."],
    ["Assistant turns indexed", "\"What did you recommend…\" is only answerable from the assistant's side."],
    ["Question routing", "Counting, totalling and ordering questions get 8 sessions in 24k characters; the rest 5 in 12k."],
    ["Dates done in code", "Each excerpt carries its date and \"N days before the question\"; the latest is marked [most recent]."],
    ["Pure Python stdlib", "No GPU, no external LLM or embedding API. A 200-conversation index builds in under a second."],
  ],

  citations: [
    "Reactor's [E#] labels on the FINAL ANSWER line",
    "else the labels in the last paragraph of its reasoning",
    "else the excerpts that contain its answer, if they single out at most half",
    "never the top-ranked session by default",
  ],
  citationNote: "Organisers' ruling: fallback citations must support the answer.",
  abstention: "Reactor answers INSUFFICIENT_EVIDENCE → we abstain, with no citations.",

  efficiency: [
    { value: "8.2k", label: "median characters sent per test question (max 14.1k)" },
    { value: "~200×", label: "less than the full history, per question" },
    { value: "19–23×", label: "less per history than loading it whole (the baseline)" },
  ],

  // Retrieval per question type, dev, v1 (README section 9).
  byType: [
    { label: "knowledge-update", a: 1.000, b: 0.819 },
    { label: "multi-session", a: 0.983, b: 0.430 },
    { label: "single-session-assistant", a: 1.000, b: 0.554 },
    { label: "single-session-user", a: 0.984, b: 0.875 },
    { label: "temporal-reasoning", a: 0.953, b: 0.315 },
  ],
  byTypeNames: ["hit@5", "answer in evidence"],
  byTypeNote: "Multi-session and temporal answers are spread over several turns: that's where v2 and a reranker aim.",

  // Headline bars (README sections 1 and 9). v1 = the version used on test.
  results: [
    { label: "dev · hit@5", value: 0.982 },
    { label: "dev-large · hit@5", value: 0.935 },
    { label: "dev · gold sessions in evidence sent", value: 0.954 },
    { label: "dev-large · gold sessions in evidence sent", value: 0.878 },
    { label: "Reactor dev sample (11 q) · evidence F1", value: 0.680 },
    { label: "Test · answer accuracy", value: null },
  ],
  resultsNote: "dev-large: ~500 conversations per user, 130 questions. Test scores come from the organisers.",

  v2: [
    ["Focused trimming", "A long turn keeps the stretch with the most question words, not its beginning."],
    ["Lead-session budget", "A clearly leading session gets up to half the budget."],
    ["Time sentences", "\"Last month\", \"three weeks ago\" pulled in from anywhere in a sent session."],
    ["Missing-words hint", "Fires on 50% of unanswerable dev questions vs 12.5% of answerable ones."],
  ],
  // Answer present in the evidence sent, v1 -> v2 (README section 7).
  ablation: [
    { label: "dev", before: 0.542, after: 0.556 },
    { label: "dev-large", before: 0.484, after: 0.492 },
  ],
  v2Note: "Built and measured on dev; not used for the test predictions (no Reactor time left). Tried and dropped: removing a session's weaker second hit lowered answer-in-evidence.",

  rules: [
    ["Reactor answers", "Final answers come only from Reactor, through the organisers' client. No hand edits."],
    ["One history per user", "A fresh Reactor history per dataset history, never reset mid-history."],
    ["Reliable runs", "600 s timeout, one retry in the same history, late answers recovered by their ref."],
    ["Reproducible", "Evidence logged per question; v1 and v1.1 drivers send byte-identical evidence on all 526 dev questions."],
  ],

  run: [
    "python run.py --questions test/questions.jsonl --histories test/histories.jsonl --out results-test.jsonl …",
    "python make_predictions.py results-test.jsonl --questions test/questions.jsonl --out predictions.jsonl",
    "streamlit run app.py   # the Recall demo: evidence timeline, citations, results",
  ],

  next: [
    "A local reranker on the BM25 candidates, for questions worded differently from the chat",
    "Convert \"last month\" and \"three weeks ago\" to dates in code",
    "Abstention from the missing-words signal, not only Reactor's",
    "Run v2 on test once there is Reactor time",
  ],
};
