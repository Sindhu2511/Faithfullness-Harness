# Benchmark Schema

Every sample in `outputs/benchmark/<run-name>/benchmark_dev.jsonl` and `benchmark_test.jsonl`
is one JSON object per line with exactly these fields, in this order. A field that does not apply
is `null`, never absent. Written by `src/rag/build_benchmark.py`.

## Fields

| Field | Type | Meaning | Example |
|-------|------|---------|---------|
| sample_id | string | Unique id: `<source_id>__<error_type>` | `"a1b2c3__broken_chain"` |
| split | string | `"dev"` or `"test"` | `"test"` |
| source_id | string | 2WikiMultiHopQA question id the sample comes from | `"a1b2c3"` |
| question_type | string | Question type: comparison, bridge_comparison, compositional or inference | `"compositional"` |
| question | string | The question | `"Where was the director of film X born?"` |
| gold_answer | string | Gold short answer from the dataset | `"April 13, 1944"` |
| answer | string | The answer to verify: the injected answer, or the original LightRAG answer for clean samples | `"The director of To Sleep with Anger was born on May 18, 1953."` |
| error_type | string | One of the 5 error types, or `"none"` for clean samples | `"broken_chain"` |
| expected_verdict | string | `"supported"`, `"contradicted"` or `"unsupported"` | `"contradicted"` |
| sentence_index | int or null | Index of the edited sentence, or of the inserted sentence for fabricated_fact; null for clean | `0` |
| original_sentence | string or null | Sentence before editing; null for fabricated_fact (inserted) and clean | `"The director of To Sleep with Anger was born on April 13, 1944."` |
| new_sentence | string or null | Edited or inserted sentence; null for clean | `"The director of To Sleep with Anger was born on May 18, 1953."` |
| original_triple | list of 3 strings or null | Gold claim triple before editing; null for fabricated_fact and clean | `["To Sleep with Anger", "director -> date of birth", "April 13, 1944"]` |
| new_triple | list of 3 strings or null | Claim triple after editing; path claims join relations with `" -> "`; null for clean | `["To Sleep with Anger", "director -> date of birth", "May 18, 1953"]` |
| path_edges | list of int or null | Edge indexes of the original path (broken_chain only) | `[0, 1]` |
| wrong_intermediate | string or null | Entity whose last-hop object was used (broken_chain only); null if the object came from the pool | `"Zdeněk Troška"` |
| reference_graph | object | Directed gold graph for the question: `nodes` (key, names), `edges` (index, subject, relation, object, subject_key, object_key), `alias_merges` | `{"nodes": [...], "edges": [...], "alias_merges": ["charles burnett"]}` |
| provenance | object | `run_name`, `llm_model`, `embed_model`, `seed` (build seed), `created` (ISO date) | `{"run_name": "pilot", "llm_model": "qwen2.5:3b", "embed_model": "nomic-embed-text", "seed": 42, "created": "2026-10-27"}` |

## Error types and expected verdicts

Matches `docs/taxonomy.md`, section 5.

| error_type | expected_verdict |
|------------|------------------|
| relation_reversal | contradicted |
| wrong_relation | contradicted |
| entity_swap | contradicted |
| broken_chain | contradicted |
| fabricated_fact | unsupported |
| none (clean sample) | supported |

## How the splits are made

- Splits are made by question, not by sample: every sample from one `source_id` is in the same
  split, so no question leaks between dev and test.
- The distinct source ids are sorted, shuffled with `--seed` (default 42), and the first
  `--dev-fraction` of them (default 0.3) go to dev; the rest go to test.
- `benchmark_summary.json` records the leakage check (overlap of source ids between splits must
  be empty).
