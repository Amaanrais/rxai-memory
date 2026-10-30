// Everything the deck shows lives here. Change a number or a line, reload index.html.
// Metrics: 0..1 values; null = "pending" (shown as not yet measured).
const DECK = {
  title: "Finding the needle",
  subtitle: "Memory retrieval for Reactor: BM25, focused evidence, answer-backed citations",
  event: "RxAI Memory Retrieval Challenge · October 2026",
  team: ["Amaan Rais (captain)", "Dedar Alam", "Frensi Qatipi"],
  repo: "rxai-memory",

  problem: {
    stats: [
      { value: "200", label: "conversations per user (test)" },
      { value: "~10k", label: "characters per conversation" },
      { value: "16k", label: "characters Reactor accepts per question" },
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

  // The pipeline diagram, left to right.
  pipeline: [
    { name: "Index", detail: "BM25 over turns, 1,500-char windows" },
    { name: "Rank", detail: "session = best chunk + ¼ next two" },
    { name: "Route", detail: "aggregate: 8 sessions / 24k; else 5 / 12k" },
    { name: "Evidence", detail: "focused excerpts, dates, [most recent]" },
    { name: "Reactor", detail: "answers with [E#] labels" },
    { name: "Cite", detail: "labels, else answer-supported excerpts" },
  ],

  retrieval: [
    ["Pure Python stdlib", "No GPU, no model downloads: one BM25 index per history, built in milliseconds."],
    ["Question routing", "Counting, totals, ordering questions get more sessions and a bigger budget."],
    ["Focused trimming", "A long turn keeps the stretch with the most question words, not its first 1,500 chars."],
    ["Leading session", "When the top session clearly wins, it gets up to half the budget."],
    ["Time made explicit", "Every excerpt says \"N days before the question\"; the latest is marked [most recent]."],
  ],

  prompting: [
    ["Work it out", "Aggregate questions reason step by step before the FINAL ANSWER line."],
    ["Latest value wins", "Update questions are told the most recent date wins."],
    ["Missing-words hint", "Rare question words absent from all evidence are named, so near misses abstain."],
    ["One value only", "Single-fact questions: no hedging, no lists (the judge counts those wrong)."],
  ],

  citations: [
    "The [E#] labels on Reactor's FINAL ANSWER line",
    "else the labels in its last reasoning paragraph",
    "else the excerpts whose text contains the answer",
    "else nothing: never the top-ranked session by default",
  ],
  citationNote: "Organisers' ruling: fallback citations must support the answer.",

  // Bar chart on the results slide.
  results: [
    { label: "dev · hit@5", value: 0.98 },
    { label: "dev-large · hit@5", value: 0.95 },
    { label: "dev-large · hit@1", value: 0.78 },
    { label: "dev-large · citation F1 (retrieval only)", value: 0.68 },
    { label: "Reactor sample (10 q) · evidence F1", value: 0.68 },
    { label: "Full run · answer accuracy", value: null },
  ],
  resultsNote: "dev-large = 500 conversations per user, the closest public split to test scale.",

  // Before -> after for the v2 evidence change (answer present in the evidence sent).
  ablation: [
    { label: "dev", before: 0.542, after: 0.550 },
    { label: "dev-large", before: 0.484, after: 0.492 },
  ],

  next: [
    "Full Reactor run on dev-large, then the 146 test questions",
    "Cross-encoder reranker on the top sessions for near-miss conversations",
    "Dense retrieval lane fused with BM25 (RRF)",
    "Tune abstention on dev-large unanswerables",
  ],
};
