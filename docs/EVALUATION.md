# GroundedAI Evaluation

## Purpose and scope

Evaluation turns retrieval and grounded generation into inspectable, reproducible behavior rather
than a manual demo. Phases 6 and 7 measure source retrieval, deterministic required-fact coverage,
citation provenance, abstention safety, latency, and available token use. It does not use an LLM
judge and does not claim that every generated sentence is entailed by its citation.

## Golden dataset

`evals/datasets/golden_v1.json` is immutable as version `1.0.0`. Material changes to cases,
fixtures, or expectations require `golden_v2`; the existing name/version must not be silently
repurposed. The dataset has 30 cases: 20 answerable and 10 unanswerable. It covers direct facts,
paraphrases, source disambiguation, one multi-document question, PDF page-labelled evidence, and
DOCX heading-labelled evidence.

All five documents are repository-owned synthetic business policies:

- `employee-handbook.pdf` - three text pages covering flexible work, leave, and security;
- `expense-policy.docx` - Travel, Meals, and Equipment heading sections;
- `refund-policy.md`;
- `operations-guide.txt`;
- `vendor-policy.md`.

Expected facts contain one or more acceptable phrases. Matching case-folds text, replaces
punctuation with spaces, and collapses whitespace; it is deterministic substring matching, not
semantic similarity. Expected sources are stable fixture IDs. Fixture preparation reuses existing
ready documents by SHA-256 checksum, maps their database UUIDs to fixture IDs for the run, and
restricts retrieval to that UUID set. It neither accumulates duplicate fixture rows nor deletes
unrelated documents. Fixture documents are metadata-tagged and excluded from ordinary retrieval;
the explicit evaluation scope is the only path that includes them.

## Modes

`retrieval_only` calls `RetrievalService.retrieve()` and never invokes generation. Answer,
citation, and abstention aggregates are reported as unavailable in this mode.

`end_to_end` calls `GroundedAnswerService.query()`. Retrieval metrics are calculated from the
trusted snapshot in the resulting persisted `QueryRun`; evaluation does not perform a second
retrieval. This mode measures answer, citation, and abstention behavior in addition to retrieval.

## Metrics

Retrieval metrics are calculated over answerable cases:

- Hit@K is 1 when any expected source occurs in the first K returned chunks, otherwise 0.
- Recall@K is the number of distinct expected sources retrieved divided by expected sources.
- MRR is the reciprocal rank of the first expected source.
- Zero-result rate and average retrieved-result count use all cases.

Fact coverage is matched expected facts divided by expected facts. The aggregate is micro-averaged
over answerable cases, so a two-fact case contributes two expectations. Exact normalized phrase
coverage is intentionally conservative and should not be relabelled as semantic correctness.

Citation validity is validated citations divided by citations returned. Source citation precision
is citations resolving to expected sources divided by citations returned. Source citation recall is
distinct expected-source matches divided by expected sources. These metrics prove citation
existence, provenance, and expected-source alignment, not claim-level entailment.

Abstention treats an answerable case as an expected answer and an unanswerable case as an expected
abstention. Correct answers, correct abstentions, false answers, and false abstentions form the
confusion matrix. Abstention precision is correct abstentions divided by all abstentions;
abstention recall is correct abstentions divided by all unanswerable cases. False-answer rate is
false answers divided by unanswerable cases and is the primary safety indicator.

Latency reports mean, p50, nearest-rank p95, and maximum in milliseconds for retrieval, generation,
and total query time. Local deterministic timings are operational observations, not production
capacity claims. Token totals and mean tokens per answered query are calculated only from reported
provider usage. Deterministic mode reports them as unavailable. Cost is null because no versioned,
model-matching pricing table is configured; deterministic execution is never presented as a
production API cost.

## Interfaces

Run from the repository root against a migrated database:

```bash
python -m scripts.evaluate --dataset golden_v1 --mode end_to_end
python -m scripts.evaluate --dataset golden_v1 --mode retrieval_only --json
```

The API provides `POST /api/v1/evaluations/run`,
`GET /api/v1/evaluations/runs/{run_id}`, and
`GET /api/v1/evaluations/runs/{run_id}/cases`. Dataset names come from a fixed registry; arbitrary
paths are rejected. The CLI and API both construct and call the same `EvaluationService`.

## Deterministic baseline

Executed at `2026-09-21T17:56:49Z` with run ID
`364d577e-90b3-42e4-9967-19af43ac5d7c`:

- Dataset: `golden_v1` version `1.0.0`, 30 cases
- Mode: `end_to_end`
- Embeddings: `deterministic-feature-hashing` / `feature-hashing-v1` / version `1`
- Generation: `deterministic` / `extractive-v1`
- Retrieval: K=5, minimum score 0.15
- Prompt: `grounded_answer_v1`
- Hit@5: 1.000
- Recall@5: 1.000
- MRR: 1.000
- Mean fact coverage: 0.9524 (20/21 facts)
- Citation validity: 1.000
- Source citation precision: 0.7407
- Source citation recall: 0.9524
- Correct answers: 20
- Correct abstentions: 3
- False answers: 7
- False abstentions: 0
- Abstention accuracy: 0.7667
- Abstention precision: 1.000
- Abstention recall: 0.3000
- False-answer rate on unanswerable cases: 0.7000
- Total latency: mean 3.626 ms, p50 3.068 ms, p95 3.870 ms, max 17.085 ms
- Tokens and cost: unavailable

The baseline is intentionally not a CI quality gate. It establishes behavior for future run-to-run
comparison; `compare_aggregates()` reports deltas for Hit@K, MRR, fact coverage, citation precision,
citation recall, false-answer rate, false-abstention rate, and p95 total latency without asserting
arbitrary thresholds. This run is the frozen Phase 6 reference and has not been rewritten or
deleted.

## Failure analysis

Retrieval found every expected source for answerable cases, but the deterministic extractive
generator cites only its first chunk. The multi-document case `multi_001` therefore matched one of
two facts and cited one of two expected sources, causing the fact and source-citation recall gaps.

Seven unanswerable cases (`unknown_001`, `unknown_002`, `unknown_004`, `unknown_006`,
`unknown_007`, `unknown_008`, and `unknown_010`) retrieved weak lexical-overlap evidence above the
0.15 threshold. The deterministic provider then extracted that first chunk, creating false answers.
This is an important measured limitation of the lexical feature-hashing baseline and current
evidence threshold, not a dataset error. Three unanswerable cases correctly returned no evidence
and abstained.

## Phase 7 calibration

Case-level analysis of the frozen baseline found one dominant safety failure: all seven false
answers were caused by irrelevant lexical overlap whose top score exceeded the retrieval threshold
of 0.15. There were no false abstentions. No answerable case had a retrieval miss. The only missing
fact was `multi_001`, where the first-chunk-only deterministic generator ignored the second
document. Answerable cases had no wrong or extra expected-source citations; the aggregate citation
precision loss came from citations attached to the seven false answers.

Top-score distributions from the frozen run were:

| Outcome group | Cases | Top-1 min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| Answerable, expected source retrieved | 20 | 0.200000 | 0.390922 | 0.389105 | 0.649519 |
| Unanswerable false answer | 7 | 0.162221 | 0.250000 | 0.231197 | 0.258199 |
| Unanswerable correct abstention | 3 | no result | no result | no result | no result |

Among answerable cases with a second result, mean top-2 score was 0.223279 and mean top-1/top-2
gap was 0.190104. Only one false-answer case had a second result; its gap was 0.009437. A score-gap
rule was not added because a top-score threshold alone addressed the measured unsafe cases.

`python -m scripts.calibrate` replays evidence decisions from immutable persisted retrieval
snapshots. It does not invoke generation, mutate the baseline, or persist one noisy evaluation run
per candidate. The machine-readable artifact is `evals/results/phase7_calibration.json`.

```bash
python -m scripts.calibrate \
  --run-id 364d577e-90b3-42e4-9967-19af43ac5d7c \
  --thresholds 0.15 0.18 0.20 0.22 0.24 0.25 0.251 0.259 0.26 0.28 0.30 \
  --output evals/results/phase7_calibration.json
```

The frozen-run sweep showed this safety/coverage tradeoff (fact coverage uses the Phase 6
first-source generator snapshot):

| Evidence minimum | False answers | False abstentions | False-answer rate | Fact coverage |
|---:|---:|---:|---:|---:|
| 0.15 | 7 | 0 | 0.700 | 0.952 |
| 0.18 | 6 | 0 | 0.600 | 0.952 |
| 0.22 | 5 | 1 | 0.500 | 0.905 |
| 0.24 | 4 | 1 | 0.400 | 0.905 |
| 0.25 | 4 | 3 | 0.400 | 0.810 |
| 0.251 | 1 | 3 | 0.100 | 0.810 |
| 0.259 | 0 | 3 | 0.000 | 0.810 |
| 0.26 | 0 | 3 | 0.000 | 0.810 |
| 0.28 | 0 | 3 | 0.000 | 0.810 |
| 0.30 | 0 | 6 | 0.000 | 0.667 |

Pareto reporting uses false answers and false abstentions directly; it does not hide the tradeoff
inside a weighted score. Threshold 0.26 was selected as a readable boundary just above the maximum
observed false-answer score of 0.258199. It is not claimed to be universally optimal.

`evidence_policy_v1` is evaluated after retrieval and before generation. It uses the top evidence
score with an inclusive minimum (`score >= minimum`). No results produce
`no_retrieval_evidence`; weak results produce `insufficient_evidence_strength`; both avoid the
generation call, persist zero generation milliseconds, and return no citations. Provider-level
abstention remains a separate category. Query and evaluation runs persist the policy version,
threshold, top score, sufficiency decision, and abstention category.

Calibration is keyed to embedding provider/model/version. The 0.26 default applies only to
`deterministic-feature-hashing` / `feature-hashing-v1` / version `1`. An OpenAI embedding profile
has no default threshold and fails safe until it receives an explicit configuration; such a value
is an operator override, not a validated calibration.

The deterministic generator was advanced from `extractive-v1` to `extractive-v2`. For a conjoined
question it extracts the leading retrieved passage plus the leading passage from each additional
document, in retrieval order, and cites exactly those used aliases. This small offline behavior
exercises multi-document context without pretending to be an LLM. It changed `multi_001` from one
of two facts/sources to two of two. It does not use evaluation case IDs, expected facts, or fixture
names.

### Measured post-calibration run

Run `b9759228-6499-4b89-86b1-14659f6eade5` completed at
`2026-09-21T19:12:09Z` with deterministic embeddings, `extractive-v2`, K=5, retrieval minimum
0.15, `evidence_policy_v1`, and evidence minimum 0.26:

| Metric | Phase 6 baseline | Phase 7 run | Delta |
|---|---:|---:|---:|
| Hit@5 | 1.0000 | 1.0000 | 0.0000 |
| Recall@5 | 1.0000 | 1.0000 | 0.0000 |
| MRR | 1.0000 | 1.0000 | 0.0000 |
| Fact coverage | 0.9524 | 0.8571 | -0.0952 |
| Citation precision | 0.7407 | 1.0000 | +0.2593 |
| Citation recall | 0.9524 | 0.8571 | -0.0952 |
| False-answer rate | 0.7000 | 0.0000 | -0.7000 |
| False-abstention rate | 0.0000 | 0.1500 | +0.1500 |
| Total latency p95 | 3.870 ms | 3.692 ms | -0.178 ms |

The new confusion matrix is 17 correct answers, 10 correct abstentions, zero false answers, and
three false abstentions. The remaining false abstentions are `backup_001` (top score 0.200000),
`backup_002` (0.243432), and `incident_001` (0.248069). Each retrieved the correct source, but its
lexical score fell below the safety threshold. There are no remaining false answers. This is the
explicit cost of the selected safety policy; improving it requires better evidence signals or
semantic embeddings, not changing golden expectations.

## Phase 7.1 frozen holdout validation

`holdout_v1` version `1.0.0` contains 14 previously unused questions over the unchanged five
synthetic fixture documents: nine answerable and five unanswerable. It includes three PDF-source
questions, two DOCX-section questions, Markdown and text questions, one multi-document question,
low-overlap paraphrases, partial-vocabulary unanswerable questions, and one no-overlap
unanswerable question. Its questions and expectations were written and schema-validated before
the first complete evaluation. They were not changed after results were observed.

The first frozen run, `b59d22d8-bb6a-4497-ac85-b60d7ed74fa1`, completed at
`2026-09-22T07:41:49Z`. It used the already-selected configuration without recalibration:

- Embeddings: `deterministic-feature-hashing` / `feature-hashing-v1` / version `1`
- Retrieval: exact cosine, K=5, minimum score 0.15
- Evidence: `evidence_policy_v1`, minimum score 0.26
- Generation: `deterministic` / `extractive-v2`
- Prompt: `grounded_answer_v1`

Measured holdout results:

- Hit@5: 0.5556
- Recall@5: 0.5000
- MRR: 0.5556
- Fact coverage: 0.3000 (3/10 facts)
- Citation validity: 1.0000
- Source citation precision: 0.5000
- Source citation recall: 0.3000
- Correct answers: 3
- Correct abstentions: 2
- False answers: 3
- False abstentions: 6
- False-answer rate: 0.6000
- False-abstention rate: 0.6667
- Total latency: mean 7.566 ms, p50 4.318 ms, p95/max 49.742 ms
- Retrieval latency: mean 7.438 ms, p50 4.189 ms, p95/max 49.576 ms
- Generation latency: mean 0.024 ms, p50 0.000 ms, p95/max 0.087 ms

The single high local latency observation occurred in this small deterministic run and is not a
production-capacity measurement.

### Calibration versus holdout

| Metric | `golden_v1` calibrated | `holdout_v1` | Holdout minus calibration |
|---|---:|---:|---:|
| Hit@5 | 1.0000 | 0.5556 | -0.4444 |
| Recall@5 | 1.0000 | 0.5000 | -0.5000 |
| MRR | 1.0000 | 0.5556 | -0.4444 |
| Fact coverage | 0.8571 | 0.3000 | -0.5571 |
| Citation precision | 1.0000 | 0.5000 | -0.5000 |
| Citation recall | 0.8571 | 0.3000 | -0.5571 |
| False-answer rate | 0.0000 | 0.6000 | +0.6000 |
| False-abstention rate | 0.1500 | 0.6667 | +0.5167 |
| Total latency p95 | 3.692 ms | 49.742 ms | +46.050 ms |

This comparison separates calibration performance from holdout performance. The frozen threshold
removed false answers on the calibration questions but did not generalize to partial-vocabulary
unanswerable questions, while lexical retrieval also missed several valid paraphrases.

### Holdout failures

Every false answer:

- `holdout_unknown_001`, “How many remote-work days may contractors use?” — top score 0.357217
  against handbook page 1. The extracted employee policy does not establish contractor coverage.
- `holdout_unknown_002`, “What is the reimbursement limit for hotel rooms on international
  travel?” — top score 0.307729 against the Meals section, followed by Travel at 0.290129. Neither
  passage supplies a hotel limit.
- `holdout_unknown_003`, “How frequently are customer data backups tested?” — top score 0.323970
  against the vendor data-protection passage. It does not state backup testing frequency.

Every false abstention:

- `holdout_equipment_001` — no qualifying retrieval result for the home-office purchase threshold.
- `holdout_leave_001` — correct handbook page retrieved at 0.238145, below evidence minimum 0.26.
- `holdout_refund_001` — no qualifying result for the made-to-order/personalized-product paraphrase.
- `holdout_remote_001` — no qualifying result for the low-overlap remote-work paraphrase.
- `holdout_security_001` — correct handbook page retrieved at 0.162221, below evidence minimum
  0.26.
- `holdout_travel_001` — no qualifying result for the seating-tier/economy-class paraphrase.

The multi-document case answered rather than abstaining, but retrieved only the expense-policy
source and covered one of two facts. This contributes to retrieval recall, fact coverage, and
citation recall loss without being an abstention-classification error.

No implementation bug was identified. The failures are consistent with the documented lexical
embedding limitation and with a top-score-only sufficiency policy: high lexical overlap cannot
prove that evidence answers the precise subject, while paraphrases with little shared vocabulary
may never reach the policy. No tuning or policy change was made.

This is a small synthetic generalization check, not a statistically representative production
validation. It measures only this corpus, these fourteen questions, deterministic embeddings, and
the deterministic extractive provider.

## Limitations

The deterministic embedding provider is lexical and its threshold is calibrated on this small
synthetic corpus. The deterministic generator remains a deliberately limited extractive test
double. Phrase matching cannot validate paraphrased facts that omit all accepted phrases.
Citation metrics do not prove semantic entailment or detect every unsupported clause. Human review,
an NLI model, or a carefully governed LLM judge may be evaluated later, but none is implemented in
Phase 7. OpenAI embedding calibration and real-provider evaluation were not executed because
`OPENAI_API_KEY` is unavailable. The deterministic threshold must not be represented as validated
for OpenAI embeddings.
