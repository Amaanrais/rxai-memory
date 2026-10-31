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
    { name: "Route", detail: "count / order / time / names: 8 sessions, 24k; else 5, 12k" },
    { name: "Retrieve", detail: "best turns ± 1, dates, N days before, [most recent]" },
    { name: "Reactor", detail: "answers from [E1]..[En], labels on the answer line" },
    { name: "Cite / abstain", detail: "labels, else support check; INSUFFICIENT_EVIDENCE → abstain" },
  ],

  design: [
    ["Turn-level BM25", "A session is ~10k characters but the fact is usually in one turn. Sessions score their best turn + 0.25 × the next two."],
    ["Assistant turns indexed", "\"What did you recommend…\" is only answerable from the assistant's side."],
    ["Question routing", "Counting, ordering and (v3) name questions get 8 sessions in 24k characters; the rest 5 in 12k."],
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
    { value: "11.3k", label: "median characters sent per test question (v3; max 21.0k)" },
    { value: "147×", label: "less than the full history, per question (median)" },
    { value: "15×", label: "less per history than loading it whole (median; 11–25×)" },
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

  // Headline bars (README sections 1 and 9). Retrieval rows: v1, the base v3 builds on. Test rows: v3, the submission.
  results: [
    { label: "dev · hit@5", value: 0.982 },
    { label: "dev-large · hit@5", value: 0.935 },
    { label: "dev · gold sessions in evidence sent", value: 0.954 },
    { label: "dev-large · gold sessions in evidence sent", value: 0.878 },
    { label: "Reactor dev sample (11 q) · evidence F1", value: 0.680 },
    { label: "Test (v3) · answered with sources, 123 of 146", value: 123 / 146 },
    { label: "Test (v3) · answer accuracy", value: null },
  ],
  resultsNote: "dev-large: ~500 conversations per user, 130 questions. Test: 23 of 146 abstained (16%; v1: 31, 21%). Accuracy comes from the organisers.",

  // v2 then v3: the submitted version (README section 7).
  v2: [
    ["v2: sharper evidence", "Focused trimming, more room for a clear leader, time sentences, work-it-out prompt for counting."],
    ["v3: name questions", "Synonyms (dog → puppy, pet), ×1.5 for a name near the entity word, 8 sessions."],
    ["Lenient abstain line", "\"INSUFFICIENT_EVIDENCE only if nothing addresses the question.\" Test abstentions 21% → 16%."],
    ["Hint off", "v2's missing-words hint flagged 12.5% of answerable dev questions, so the submission leaves it out."],
  ],
  // Answer present in the evidence sent, v2 -> v3, on the 12 answerable name questions (README section 7).
  ablation: [
    { label: "dev · names", before: 0.917, after: 0.917 },
    { label: "dev-large · names", before: 0.833, after: 0.917 },
  ],
  v2Note: "Other questions get exactly v2's evidence (dev 0.542 → 0.556 from v1 to v2). 48 of 146 test questions ask for a name.",

  rules: [
    ["Reactor answers", "Final answers come only from Reactor, through the organisers' client. No hand edits."],
    ["One history per user", "A fresh Reactor history per dataset history, never reset mid-history."],
    ["Reliable runs", "600 s timeout; on a timeout, listen up to 120 s for the late answer before one retry. One prediction per question, unit-tested."],
    ["Reproducible", "Evidence logged per question; tag final-submission. The test driver sends byte-identical questions to run_v3.py, unit-tested."],
  ],

  run: [
    "python run_v3_test.py --questions test/questions.jsonl --histories test/histories.jsonl --out results-test-v3.jsonl …",
    "python make_predictions.py results-test-v3.jsonl --questions test/questions.jsonl --out predictions.jsonl",
    "streamlit run app.py   # the Recall demo: evidence timeline, citations, results",
  ],

  next: [
    "A local reranker on the BM25 candidates, for questions worded differently from the chat",
    "Convert \"last month\" and \"three weeks ago\" to dates in code",
    "Abstention from the missing-words signal, not only Reactor's",
    "Check how many traps the lenient line lets through; bring a stricter hint back if needed",
  ],
};
