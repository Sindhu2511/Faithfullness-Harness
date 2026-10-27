# Faithfulness Evaluation Harness for Graph-Based RAG

## 1. Overview

A system that checks whether answers produced by a Graph-RAG system are actually supported by a
knowledge graph. Each answer is split into individual factual claims, each claim is written as a
(subject, relation, object) triple and checked against a directed reference graph, and every
error is given a specific type (for example relation reversal or broken multi-hop chain) with an
explanation. It is meant for developers and researchers who build Graph-RAG systems and need to
know not only whether an answer is wrong, but how and where it went wrong.

## 2. Team and guide

**Team 17, Department of Information Technology, BVRIT Hyderabad College of Engineering for Women**

K. Sree Sindhu (23WH1A1251), V. Malleswari (23WH1A1229), M. Jahnavi (23WH1A1242), S. Sahasra (23WH1A1203)

Guide: Ms. Suparna Das, Assistant Professor, Department of CSE

## 3. Problem

Existing hallucination checkers do not tell a Graph-RAG developer what kind of error was made.
GraphEval and RefChecker check claim triples against plain text, and GGA uses model internals
(attention and hidden states) to give a whole-answer label. None of them names error types such
as relation reversal, entity swap or a broken multi-hop chain (see
[docs/taxonomy.md](docs/taxonomy.md), section 6).

The graph built by LightRAG cannot fill this gap on its own: LightRAG 1.5.7 stores its graph
undirected, so (A, r, B) and (B, r, A) look identical and relation reversal cannot be detected
from it. See [docs/decisions.md](docs/decisions.md), D1.

## 4. Novelty

- **Error taxonomy:** five Graph-RAG hallucination types (relation reversal, wrong relation,
  entity swap, broken multi-hop chain, fabricated fact), each with a formal rule over triples and
  a fixed decision order.
- **Structure-first hybrid verifier:** symbolic checks against the directed reference graph
  (edge, direction, path) first, with NLI only as a fallback.
- **Explainable claim-level diagnosis:** each claim gets a verdict, an error type and the
  evidence edge or path behind it.
- **Benchmark:** an auto-labelled perturbation benchmark built by injecting each error type into
  correct answers, plus about 100 human-labelled real answers.

## 5. Objectives

1. Define a Graph-RAG hallucination taxonomy and benchmark
2. Develop a hybrid structure-aware verifier (symbolic checks against the reference knowledge
   graph first, NLI as fallback)
3. Generate explainable claim-level diagnosis
4. Evaluate against RAGAS, RefChecker, MiniCheck and NLI-only, and validate against human labels

## 6. Project status

| Phase | Task | Status |
|-------|------|--------|
| Phase A Taxonomy | A1-A3 definitions, examples, triple changes | Drafted ([docs/taxonomy.md](docs/taxonomy.md)) |
| Phase A | A4 comparison with GraphEval, RefChecker, GGA | Done ([docs/taxonomy.md](docs/taxonomy.md) section 6) |
| Phase A | A5 guide sign-off | Pending |
| Phase B Source answers | B1-B2 question selection (300 questions, 75 per type) | Done |
| Phase B | B3 LightRAG answers | Script done, 20-question pilot in progress |
| Phase B | B4 correct-answer filter | Script done, self-test passing, waiting for pilot |
| Phase B | B5 sentence-to-edge mapping | Script done, self-test passing (9/9), waiting for pilot |
| Phase C Error injection | C1-C3 injectors for all 5 error types (relation reversal, entity swap, wrong relation, fabricated fact, broken chain) with taxonomy-consistency check | Scripts done, self-test 22 cases, waiting for pilot |
| Phase C | C4 spot-check sheet | Script done (spotcheck.csv), human review pending |
| Phase C | C5 balancing per error type plus clean samples | Script done, self-test 6 cases, waiting for pilot |
| Phase D Human-annotated set | D1-D5 | Not started |
| Phase E Packaging | E1 fixed schema | Done ([docs/benchmark_schema.md](docs/benchmark_schema.md)) |
| Phase E | E2 dev/test split by question | Script done, waiting for pilot |
| Phase E | E3 README stats table | Pending (after full run) |
| Objectives 2-4 | Verifier, diagnosis, evaluation | Not started |

## 7. Key design decisions

**D1. Gold evidence triples are the directed reference graph.** LightRAG 1.5.7 stores its graph
in an undirected `nx.Graph()` and returns relationship endpoints in alphabetical order, so its
retrieved subgraph cannot show direction. Relation reversal and entity swap both need direction.
The reference graph G for each question is therefore its 2WikiMultiHopQA gold `evidences`
triples, which are directed. LightRAG is used only to generate answers; its retrieved entities,
relations and chunks are kept as supporting evidence and for the NLI fallback. Because gold entity
names are not always consistent ("Charles Burnett" and "Charles Burnett (director)"), entities are
normalised before any path check. Full record: [docs/decisions.md](docs/decisions.md).

## 8. Repository structure

```
faithfulness-harness/
├── data/
│   ├── test_responses.json         # 6 hand-written responses with gold claims (prototype)
│   ├── questions/
│   │   ├── questions.jsonl         # 300 sampled 2WikiMultiHopQA questions (75 per type)
│   │   └── questions_summary.json  # counts, seed, split, source dataset id
│   └── lightrag/                   # LightRAG index per run (git-ignored)
├── docs/
│   ├── objective1_plan.md          # plan for the claim-extraction prototype
│   ├── taxonomy.md                 # hallucination taxonomy and comparison with existing work
│   ├── decisions.md                # design decisions (D1: source of edge direction)
│   └── benchmark_schema.md         # fixed benchmark sample schema, verdicts, split method
├── src/
│   ├── baseline.py                 # spaCy baseline: one sentence = one claim
│   ├── extractor.py                # LLM extractor: atomic claims + triples via Ollama
│   ├── evaluate.py                 # scores extractors against gold claims
│   ├── data/
│   │   └── prepare_questions.py    # B1-B2: load and sample 2WikiMultiHopQA
│   └── rag/
│       ├── run_lightrag.py         # B3: LightRAG answers + retrieved subgraph per question
│       ├── filter_correct.py       # B4: judge answers against gold, split correct/incorrect
│       ├── entities.py             # entity normalisation, reference graph, path search
│       ├── map_sentences.py        # B5: map answer sentences to gold edges and paths
│       ├── inject_errors.py        # C1-C3: plant the 5 error types into correct answers
│       └── build_benchmark.py      # C4-C5, E1-E2: balance, split by question, spot-check sheet
├── outputs/
│   ├── baseline_claims.json        # prototype: baseline claims
│   ├── llm_claims.json             # prototype: LLM extractor claims
│   ├── eval_report.json            # prototype: evaluation report
│   ├── lightrag/<run-name>/        # created by the pipeline (answers, judged, mapped, summaries)
│   └── benchmark/<run-name>/       # created by the pipeline (injected, dev/test, spot-check)
├── pyproject.toml
└── uv.lock
```

## 9. Setup (Windows PowerShell)

Prerequisites: Python 3.12 or newer (`requires-python = ">=3.12"`), [uv](https://docs.astral.sh/uv/),
[Ollama](https://ollama.com/).

```powershell
uv sync
winget install Ollama.Ollama
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Notes:
- The Hugging Face dataset (`framolfese/2WikiMultihopQA`) is public, so no login is needed.
- In Windows PowerShell 5.1, view output files with `Get-Content <file> -Encoding UTF8` so that
  accented names (for example "Zdeněk Troška") display correctly.

## 10. Running the benchmark pipeline

All commands run from the repo root.

### 10.1 Prepare questions

Purpose: load 2WikiMultiHopQA [7] and select a fixed, stratified question set (tasks B1-B2).

```powershell
uv run python src/data/prepare_questions.py
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--split` | `validation` | Dataset split: `train`, `validation` or `test` (test answers are hidden; the script stops) |
| `--n` | `300` | Questions to sample, split equally across question types |
| `--seed` | `42` | Random seed for repeatable sampling |
| `--min-evidences` | `2` | Keep only questions with at least this many gold evidence triples |
| `--out-dir` | `data/questions` | Output folder |

- Input: Hugging Face dataset `framolfese/2WikiMultihopQA` (downloaded and cached by `datasets`).
- Output: `data/questions/questions.jsonl`, `data/questions/questions_summary.json`.

### 10.2 Run LightRAG

Purpose: index the questions' context paragraphs with LightRAG and save, per question, the answer
and the retrieved entities, relations and chunk ids (task B3). Ollama must be running. Indexing
and answering are resumable: re-running skips finished documents and questions.

20-question pilot (the default):

```powershell
uv run python src/rag/run_lightrag.py
```

All 300 questions:

```powershell
uv run python src/rag/run_lightrag.py --limit 0 --run-name full
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--questions` | `data/questions/questions.jsonl` | Question file |
| `--limit` | `20` | Number of questions, in file order; `0` means all |
| `--run-name` | `pilot` | Name of the run folder |
| `--llm-model` | `qwen2.5:3b` | Ollama LLM for indexing and answering |
| `--embed-model` | `nomic-embed-text` | Ollama embedding model |
| `--embed-dim` | `768` | Embedding dimension |
| `--ollama-host` | `http://localhost:11434` | Ollama server |
| `--num-ctx` | `8192` | Ollama context window |
| `--mode` | `hybrid` | LightRAG query mode: `local`, `global`, `hybrid`, `naive`, `mix`, `bypass` |
| `--max-async` | `1` | Concurrent LLM and embedding calls |
| `--llm-timeout` | `900` | Seconds per LLM call |

- Input: `data/questions/questions.jsonl`.
- Output: index in `data/lightrag/<run-name>/` (git-ignored);
  `outputs/lightrag/<run-name>/answers.jsonl`, `outputs/lightrag/<run-name>/run_summary.json`.
- Runtime: **rough estimate only, not measured.** On CPU with `qwen2.5:3b`, indexing dominates,
  because every unique context paragraph needs LLM entity and relation extraction. A full run of
  300 questions may take one to two days. Use the pilot's `run_summary.json` to make a better
  estimate before starting the full run.

Results: Pilot in progress.

### 10.3 Filter correct answers

Purpose: judge each LightRAG answer against the gold answer. Correct answers are the clean base
for error injection (Phase C); incorrect answers go to the human-annotated set (Phase D) (task B4).

```powershell
uv run python src/rag/filter_correct.py --self-test
uv run python src/rag/filter_correct.py
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--run-name` | `pilot` | Run folder under `outputs/lightrag/` |
| `--recall-threshold` | `0.8` | Minimum gold-token recall for the token-recall rule |
| `--review-file` | none | Filled review CSV whose labels override the automatic ones |
| `--self-test` | off | Run built-in cases and exit |

Rules, in order: error, yes/no, exact containment, comparison ambiguity (answer names both
options, so it goes to human review), token recall.

Human review workflow:
1. Run the filter. Answers that name both options of a comparison question are written to
   `outputs/lightrag/<run-name>/review.csv`.
2. Open `review.csv` and fill the `human_label` column with `correct` or `incorrect`.
3. Rerun with the review file:

```powershell
uv run python src/rag/filter_correct.py --review-file outputs/lightrag/pilot/review.csv
```

- Input: `outputs/lightrag/<run-name>/answers.jsonl`.
- Output, in `outputs/lightrag/<run-name>/`: `judged.jsonl`, `clean.jsonl`, `incorrect.jsonl`,
  `review.csv`, `filter_summary.json`.

Results: Pilot in progress.

### 10.4 Map sentences to gold edges

Purpose: for each answer, build the directed reference graph from the gold evidences and record
which sentence states which gold edge or multi-hop path (task B5).

```powershell
uv run python src/rag/map_sentences.py --self-test
uv run python src/rag/map_sentences.py
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--run-name` | `pilot` | Run folder under `outputs/lightrag/` |
| `--questions` | `data/questions/questions.jsonl` | Question file (source of gold evidences) |
| `--source` | `judged` | Read `judged.jsonl` or `clean.jsonl` |
| `--max-hops` | `2` | Longest multi-hop path searched |
| `--self-test` | off | Run built-in cases and exit |

- Input: `outputs/lightrag/<run-name>/judged.jsonl` (or `clean.jsonl`),
  `data/questions/questions.jsonl`.
- Output, in `outputs/lightrag/<run-name>/`: `mapped.jsonl`, `map_summary.json`.

Results: Pilot in progress.

### 10.5 Inject errors

Purpose: plant known errors into answers labelled correct to build the auto-labelled benchmark
(tasks C1-C3). Every candidate is kept only if its new triple classifies as its own error type
under [docs/taxonomy.md](docs/taxonomy.md) rules and decision order; at most one sample per type
per answer.

```powershell
uv run python src/rag/inject_errors.py --self-test
uv run python src/rag/inject_errors.py
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--run-name` | `pilot` | Run folder under `outputs/lightrag/` (input) and `outputs/benchmark/` (output) |
| `--questions` | `data/questions/questions.jsonl` | Question file (source of the entity-swap pools) |
| `--types` | `relation_reversal,entity_swap,wrong_relation,fabricated_fact,broken_chain` | Comma list of error types to inject |
| `--seed` | `42` | Random seed |
| `--self-test` | off | Run built-in cases and exit |

What each type changes:
- **relation_reversal:** swaps the subject and object mentions of a stated person-to-person edge
  (father, mother, child, doctoral advisor); symmetric relations (spouse, sibling) are never
  reversed.
- **entity_swap:** replaces the object of a stated edge with another object of the same relation
  taken from all 300 questions, never an entity from this question's graph.
- **wrong_relation:** rewrites a one-fact sentence with a template using another relation from
  the same group (for example director to producer), keeping both entities.
- **fabricated_fact:** inserts a new template sentence giving a mentioned entity a relation it
  has no edge for, with an object not in the graph; no existing sentence is changed.
- **broken_chain:** in a sentence that implies a multi-hop path without naming the middle
  entity, replaces the end of the path with the last-hop object of a different intermediate
  entity (or, failing that, a pool object).

Skip reasons reported per type: `no_direct_edge`, `relation_not_reversible`, `span_not_found`,
`span_overlap`, `empty_pool`, `multi_fact_sentence`, `relation_not_groupable`, `unknown_kind`,
`no_mentioned_entity`, `no_path_sentence`, `no_replacement`, `invalid`, `taxonomy_mismatch`,
`no_valid_candidate`.

- Input: `outputs/lightrag/<run-name>/mapped.jsonl` (rows labelled correct),
  `data/questions/questions.jsonl`.
- Output, in `outputs/benchmark/<run-name>/`: `injected.jsonl`, `inject_summary.json`.

Results: Pilot in progress.

### 10.6 Build the benchmark

Purpose: balance error types, add clean samples, split by question and write the fixed schema
(tasks C4-C5, E1-E2). Field definitions: [docs/benchmark_schema.md](docs/benchmark_schema.md).

```powershell
uv run python src/rag/build_benchmark.py --self-test
uv run python src/rag/build_benchmark.py
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--run-name` | `pilot` | Run folder |
| `--per-type` | `0` | Samples per error type; `0` means the smallest non-zero type count |
| `--clean-count` | `0` | Clean samples; `0` means the same as the per-type target |
| `--dev-fraction` | `0.3` | Share of questions in the dev split |
| `--spotcheck-per-type` | `20` | Samples per error type in the spot-check sheet |
| `--seed` | `42` | Random seed for balancing, splitting and the spot-check |
| `--self-test` | off | Run built-in cases and exit |

- **Balancing:** each error type is sampled down to the same target. Types with 0 samples are
  left out, and types below an explicit `--per-type` are kept in full; both are reported as
  warnings.
- **Clean samples:** unmodified correct answers with `error_type` `none` and `expected_verdict`
  `supported`.
- **Split by question:** all samples from one question go to the same split, so no question
  appears in both dev and test. The summary records this leakage check.
- **Spot-check review:** open `spotcheck.csv`, fill `reads_naturally` and `label_correct` for
  each row (and `notes` if needed).

- Input: `outputs/benchmark/<run-name>/injected.jsonl`, `outputs/lightrag/<run-name>/mapped.jsonl`
  (and `answers.jsonl` for model names).
- Output, in `outputs/benchmark/<run-name>/`: `benchmark_dev.jsonl`, `benchmark_test.jsonl`,
  `spotcheck.csv`, `benchmark_summary.json`.

Results: Pilot in progress.

### 10.7 Full pipeline in one go

Pilot (20 questions):

```powershell
uv run python src/data/prepare_questions.py
uv run python src/rag/run_lightrag.py
uv run python src/rag/filter_correct.py
uv run python src/rag/map_sentences.py
uv run python src/rag/inject_errors.py
uv run python src/rag/build_benchmark.py
```

Full run (300 questions):

```powershell
uv run python src/data/prepare_questions.py
uv run python src/rag/run_lightrag.py --limit 0 --run-name full
uv run python src/rag/filter_correct.py --run-name full
uv run python src/rag/map_sentences.py --run-name full
uv run python src/rag/inject_errors.py --run-name full
uv run python src/rag/build_benchmark.py --run-name full
```

## 11. Self-tests

```powershell
uv run python src/rag/filter_correct.py --self-test
uv run python src/rag/map_sentences.py --self-test
uv run python src/rag/inject_errors.py --self-test
uv run python src/rag/build_benchmark.py --self-test
```

Each case prints `PASS` or `FAIL`, followed by a total line. Passing looks like:

| Script | Expected total line |
|--------|---------------------|
| `filter_correct.py` | `10/10 passed` |
| `map_sentences.py` | `9/9 passed` |
| `inject_errors.py` | `22/22 passed` |
| `build_benchmark.py` | `6/6 passed` |

Passing exits with code 0; any failure gives exit code 1. None of the self-tests needs Ollama or
the dataset.

## 12. Earlier work: claim extraction prototype

This was the first prototype. It is now one component of the verifier: claim triple extraction.
It splits responses into atomic claims with (subject, relation, object) triples.

```powershell
uv run python src/baseline.py     # spaCy baseline -> outputs/baseline_claims.json
uv run python src/extractor.py    # LLM extractor  -> outputs/llm_claims.json
uv run python src/evaluate.py     # scores both    -> outputs/eval_report.json
```

The extractor uses Ollama with `qwen2.5:7b` (`ollama pull qwen2.5:7b`). Input for all three
scripts is `data/test_responses.json`.

Results (development set: 6 responses, 18 gold claims):

| Metric | spaCy baseline | LLM extractor |
|--------|----------------|---------------|
| Precision | 0.40 | 0.94 |
| Recall | 0.22 | 0.83 |
| F1 | 0.29 | 0.88 |
| Planted errors kept | 0/2 | 2/2 |
| Pronoun leaks | 4 | 0 |

Limitations of the prototype:
- The set is small, and the prompt was tuned on it, so these scores are optimistic. Held-out
  evaluation is pending.
- Count facts ("won two Nobel Prizes") and sentences linking three entities are sometimes not
  split.
- Claim matching uses character similarity, which can credit claims with unresolved pronouns. It
  will be replaced with NLI-based matching in Objective 3.

## 13. Known limitations

- LightRAG's graph is undirected, so its retrieved relations cannot show direction; the gold
  evidence triples are used as the directed reference graph instead
  ([docs/decisions.md](docs/decisions.md), D1).
- `qwen2.5:3b` sometimes produces LLM output format errors during LightRAG entity and relation
  extraction, and the affected relations are dropped from the index.
- `map_sentences.py` does not resolve pronouns or surname-only mentions, so a sentence such as
  "He was born in 1944" is not mapped to any edge.
- The comparison-question regex in `filter_correct.py` can misread a question that has an extra
  comma before the two options.
- Dates written in a different format from the gold triple cannot be located in the answer, so
  entity swap and broken chain skip them (`span_not_found`).
- Wrong relation and fabricated fact use fixed sentence templates, so the edited sentence can
  differ in style from the rest of the answer.
- A wrong-relation sample can occasionally be true in the real world (for example a director who
  also produced the film). Labels are defined against the reference graph
  ([docs/decisions.md](docs/decisions.md), D1), not world knowledge.
- The verifier (Objective 2) will need to treat inverse relations (father/mother and child) as
  equivalent.

## 14. References

1. H. Zhang, Y. Xu, L. Perez-Beltrachini, "Fine-grained Natural Language Inference Based
   Faithfulness Evaluation for Diverse Summarisation Tasks (INFUSE)," EACL 2024.
2. Z. Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation," arXiv:2410.05779,
   2024.
3. P. Laban, T. Schnabel, P. N. Bennett, M. A. Hearst, "SummaC: Re-Visiting NLI-based Models for
   Inconsistency Detection in Summarization," TACL 2022.
4. Sansford, Richardson, Petric Maretic, Nait Saada, "GraphEval: A Knowledge-Graph
   Based LLM Hallucination Evaluation Framework," KiL Workshop at KDD 2024, arXiv:2407.10793.
5. Hu et al., "RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for
   Large Language Models," arXiv:2405.14486.
6. Li et al., "Detecting Hallucinations in Graph Retrieval-Augmented Generation via Attention
   Patterns and Semantic Alignment (GGA)," arXiv:2512.09148.
7. Ho, Nguyen, Sugawara, Aizawa, "Constructing A Multi-hop QA Dataset for Comprehensive
   Evaluation of Reasoning Steps (2WikiMultiHopQA)," COLING 2020, arXiv:2011.01060.
