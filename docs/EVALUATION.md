# GroundedAI Evaluation

## Purpose and scope

Evaluation turns retrieval and grounded generation into inspectable, reproducible behavior rather
than a manual demo. Phase 6 measures source retrieval, deterministic required-fact coverage,
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
false-answer rate, and p95 total latency without asserting arbitrary thresholds.

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

## Limitations

The deterministic embedding provider is lexical and the deterministic generator is first-chunk
extractive. Phrase matching cannot validate paraphrased facts that omit all accepted phrases.
Citation metrics do not prove semantic entailment or detect every unsupported clause. Human review,
an NLI model, or a carefully governed LLM judge may be evaluated later, but none is implemented in
Phase 6. Real-provider evaluation was not executed because `OPENAI_API_KEY` is unavailable.
