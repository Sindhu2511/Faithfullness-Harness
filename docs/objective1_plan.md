# Objective 1 Plan: Graph-Native Claim Extraction

Faithfulness Evaluation Harness for Graph-Based RAG - Team 17

## 1. Goal

Turn a Graph-RAG answer into a list of atomic factual claims, where each claim is also a
(subject, relation, object) triple tied to the knowledge graph's own entities and relation types.
Prove it works on real data, against human-written answers, with a fair comparison of methods.

### What changes from the prototype

| Area | Prototype (done) | Major-project version (this plan) |
|------|------------------|-----------------------------------|
| Data | 6 responses written by us | 100 real questions from 2WikiMultiHopQA, real model answers |
| Gold claims | 1 annotator, 18 claims | 2 annotators, agreement measured |
| Triples | Free text ("born_in") | Restricted to the graph's relation list, entities linked to graph nodes |
| Methods compared | 2 | 5 extractors x 3 model sizes |
| Matching | Character similarity | Embedding + NLI matching, checked against human judgment |
| Test set | Same set used for tuning | Separate dev set (tuning) and frozen test set (reporting) |

### Why 2WikiMultiHopQA

- Multi-hop questions over Wikipedia (Ho et al., COLING 2020).
- Each question ships with gold "evidence": Wikidata triples (subject, relation, object) that form
  the reasoning path. This is the same shape as our graph, so it gives us a ready-made schema and
  gold graph facts for free.
- Each question also ships with its context paragraphs, which become the text corpus.
- Available as Parquet files on Hugging Face (dev split is enough).

---

## 2. Phases at a glance

| Phase | Output | Owner (suggested) | Time (estimate) |
|-------|--------|-------------------|-----------------|
| 1. Data and responses | 100 questions, corpus, relation schema, model answers, planted errors | Sindhu | 4-5 days |
| 2. Gold annotation | Gold claims + gold triples for 100 answers, 50 double-annotated | All 4 (split) | 4-5 days |
| 3. Extractors | 5 extractors including graph-native | Sindhu + 1 | 5-6 days |
| 4. Evaluation and ablation | Validated matcher, full results table | 2 members | 4-5 days |
| 5. Error analysis and write-up | Error taxonomy, figures, report section, README | All 4 | 3-4 days |

Total: about 3-4 weeks part-time (estimate). Phases 2 and 3 can run in parallel once Phase 1 ends.

---

## 3. Phase 1: Data and responses

### 3.1 Download and sample

1. Add dependencies: `uv add pandas pyarrow huggingface_hub`.
2. Download the 2WikiMultiHopQA dev split (Parquet) from Hugging Face.
3. Sample 100 questions with a fixed random seed (e.g. 42) so results are repeatable.
4. Balance question types (comparison, inference, compositional, bridge-comparison), about 25 each.
5. Split: **dev = 30** (allowed to tune prompts on), **test = 70** (frozen; never look at model
   output on it until final runs).

Files:
- `src/data/download.py`
- `data/raw/2wiki_dev.parquet` (not committed if large; add to `.gitignore`)
- `data/sample/questions_dev.json`, `data/sample/questions_test.json`

### 3.2 Build the corpus and relation schema

1. Corpus: save every context paragraph as `{entity_title, text}`. Paragraph titles become the
   **entity list** (graph nodes).
2. Schema: collect every relation that appears in the sampled evidence triples
   (e.g. `date of birth`, `director`, `country of citizenship`, `father`), convert to snake_case,
   and save as the allowed relation list. Add one fallback relation: `other`.
3. Gold graph facts: save all evidence triples. Objective 2 will load these into Neo4j.

Files:
- `data/corpus.json`
- `data/schema/relations.json`
- `data/schema/entities.json`
- `data/gold_graph_triples.json`

### 3.3 Generate answers (stand-in Graph-RAG)

Short answers like "Yes" or "1952" produce only one claim, which is useless for testing extraction.
So answers must explain their reasoning.

1. For each question, give the LLM (Ollama, `qwen2.5:7b`) its context paragraphs plus the matching
   graph triples.
2. Prompt it to answer in 3-6 sentences, stating each fact used.
3. Label this clearly in the report as a stand-in until the full Graph-RAG retriever is built in
   Objective 2. The extractor only needs realistic answers; it does not care how they were retrieved.

File: `src/data/generate_responses.py` -> `data/responses_{dev,test}.json`

### 3.4 Plant known errors

1. For 30% of answers, change one fact in code: swap a date, a person, or a place for another
   value of the same type taken from the corpus.
2. Record exactly what was changed (`planted_errors` field), so later objectives have ground truth
   for hallucination detection.
3. Changes are done by code, not by the LLM, so they are exact and repeatable.

File: `src/data/plant_errors.py`

**Exit check:** 100 answers exist, each 3-6 sentences, about 30 with a recorded planted error.

---

## 4. Phase 2: Gold annotation

### 4.1 Annotation guideline (1 page, shared with all annotators)

1. One fact per claim: one subject, one relation, one object.
2. Replace pronouns with full names.
3. Skip opinions and hedges.
4. Copy facts exactly as written, including wrong ones.
5. Write a triple for every claim using only relations from `relations.json`; use `other` if none
   fits. Subject and object use entity titles from `entities.json` where one exists.

Add 5 worked examples taken from the dev set.

### 4.2 Process

1. Annotators write gold claims **before** seeing any extractor output (avoids bias).
2. All 100 answers get one annotation. Split across 4 members: 25 each.
3. 50 answers (mixed dev and test) get a second independent annotation.
4. Disagreements on the 50 are discussed and resolved into one final gold version.

### 4.3 Agreement measures

| Measure | What it shows |
|---------|---------------|
| Inter-annotator claim F1 | Treat annotator A as gold and B as predictions; how closely their claim lists match |
| Cohen's kappa on relation labels | For claims both annotators wrote, how often they picked the same relation, corrected for chance |

Cohen's kappa = agreement score between two people, corrected for luck.
Kappa above 0.6 is usually described as substantial agreement.

The human agreement F1 is also the **ceiling**: no extractor should be expected to beat humans.

Files:
- `docs/annotation_guideline.md`
- `data/gold/gold_annotator{A,B}.json`, `data/gold/gold_final.json`
- `src/eval/agreement.py`

Annotation time: about 5 minutes per answer, so roughly 12 person-hours in total (estimate).

**Exit check:** gold claims + triples for all 100 answers, agreement scores computed.

---

## 5. Phase 3: Extractors

All extractors share one output format (already used in the prototype), so one evaluation script
scores all of them.

| ID | Extractor | What it tests |
|----|-----------|---------------|
| E0 | spaCy sentence split (existing `baseline.py`) | Floor: one sentence = one claim |
| E1 | Rule-based dependency split | Can grammar rules alone split "and" / "who" clauses? |
| E2 | LLM zero-shot | LLM with rules but no examples |
| E3 | LLM few-shot + code checks (existing `extractor.py`) | Current prototype |
| E4 | LLM graph-native | E3 + relation list + entity linking (the novel contribution) |

### 5.1 E1: Rule-based dependency split

Dependency parse = grammar tree showing how words connect.

1. Use spaCy's parse to split on joined verbs (`conj`) and "who/which" clauses (`relcl`).
2. Resolve "he/she/they" to the most recent person mentioned (simple rule).
3. Purpose: shows whether an LLM is actually needed, which examiners often ask.

File: `src/extract/rules.py`

### 5.2 E4: Graph-native extraction (main contribution)

1. **Relation list:** the Pydantic schema restricts `relation` to values from `relations.json`
   (built at runtime as an Enum). The model cannot invent relation names.
2. **Entity linking:** after extraction, match subject and object to `entities.json` titles:
   exact match first, then fuzzy match (`rapidfuzz`), then embedding similarity. Store the
   matched entity ID, or `null` if nothing passes the threshold.
3. **Value normalisation:** dates to `YYYY-MM-DD` or `YYYY`, numbers to digits.
4. **Flags:** keep the prototype's `triple_issues` (compound subject, object not in text, etc.).
   Add `unlinked_entity` and `relation_other`.
5. Output adds `subject_id`, `object_id` to each claim.

File: `src/extract/graph_native.py`

### 5.3 Rules for fair comparison

1. Prompts may only be tuned on the **dev** set (30).
2. Prompt examples must use entities **not** in the test set.
3. Freeze all prompts and code (git tag `obj1-frozen`) before the first test-set run.

**Exit check:** all 5 extractors run on the dev set and produce valid output files.

---

## 6. Phase 4: Evaluation and ablation

### 6.1 Better claim matcher

Problem with the prototype: character similarity gave credit to "He proposed..." and can confuse
"born in Ulm" with "born in 1879".

New matcher:
1. Shortlist candidate gold claims by embedding similarity (`sentence-transformers`,
   `all-MiniLM-L6-v2`).
2. Confirm with NLI in both directions (claim entails gold AND gold entails claim) using a
   Hugging Face NLI model such as `cross-encoder/nli-deberta-v3-base`.
3. One-to-one greedy matching, as in the prototype.

**Validate the matcher itself:** two members label 100 random (predicted, gold) pairs as
same/different by hand. Report how often the matcher agrees with them. Pick thresholds on dev only.

This same NLI model is reused in Objective 3, so the work is not wasted.

### 6.2 Metrics

| Metric | Meaning |
|--------|---------|
| Claim precision / recall / F1 | Are the extracted facts the right ones? |
| Triple accuracy | Correct subject entity + correct relation + correct object, all three |
| Entity linking accuracy | Share of subjects/objects linked to the right graph node |
| Relation accuracy | Share of matched claims with the correct relation label |
| Planted errors kept | Wrong facts that survived extraction (must be near 100%) |
| Pronoun leaks | Claims still containing he/she/they |
| Seconds per answer | Practical cost |

### 6.3 Ablation grid

Ablation = remove or swap one part to see its effect.

| Factor | Values |
|--------|--------|
| Extractor | E0, E1, E2, E3, E4 |
| Model size (E2-E4 only) | `qwen2.5:3b`, `qwen2.5:7b`, `qwen2.5:14b` |
| Code checks (E3, E4) | on / off |

Runs: 2 rule-based + (3 LLM extractors x 3 sizes) + 6 checks-off variants, on 70 test answers.

Compute cost: at about 30 s per answer for 7B on CPU, the LLM runs total roughly 6-10 hours
(estimate). 14B is slower and needs about 9 GB download and 16 GB+ RAM (estimate). Run overnight;
save results per run so a crash does not lose finished work.

Files:
- `src/eval/matcher.py`, `src/eval/metrics.py`, `src/eval/run_ablation.py`
- `outputs/runs/{extractor}_{model}_{checks}.json`
- `outputs/results_table.csv`

**Exit check:** one results table covering every row of the grid, test set only.

---

## 7. Phase 5: Error analysis and write-up

### 7.1 Error taxonomy

Label every miss and extra from the best system and the baseline into categories:

| Category | Example from prototype |
|----------|------------------------|
| Merged facts | "worked with Babbage on the Analytical Engine" |
| Dropped count or quantity | "won two Nobel Prizes" missing |
| Compound subject | "Pierre Curie and Marie Curie discovered polonium" |
| Wrong relation label | `was` instead of `occupation` |
| Unlinked entity | Name spelled differently from graph node |
| Fact "corrected" by model | Planted error silently fixed |

Report counts and percentages per category.

### 7.2 Deliverables

1. Results table (main ablation) and a bar chart of F1 by extractor.
2. Error taxonomy table with 1 example per category.
3. Report section: dataset, annotation, methods, results, errors, limitations.
4. 3-4 review slides: problem, method diagram, results table, error examples.
5. README updated with how to reproduce every number.

---

## 8. Target repository layout

```
faithfulness-harness/
├── data/
│   ├── raw/                  # downloaded parquet (gitignored)
│   ├── sample/               # 100 sampled questions, dev/test split
│   ├── schema/               # relations.json, entities.json
│   ├── gold/                 # annotator files + final gold
│   ├── corpus.json
│   ├── gold_graph_triples.json
│   └── responses_{dev,test}.json
├── docs/
│   ├── objective1_plan.md
│   └── annotation_guideline.md
├── src/
│   ├── data/                 # download, generate_responses, plant_errors
│   ├── extract/              # baseline, rules, llm, graph_native
│   └── eval/                 # matcher, metrics, agreement, run_ablation
├── outputs/
│   ├── runs/
│   └── results_table.csv
├── pyproject.toml
└── uv.lock
```

The prototype files (`baseline.py`, `extractor.py`, `evaluate.py`) move into `src/extract/` and
`src/eval/` in Phase 3.

---

## 9. Risks and fixes

| Risk | Fix |
|------|-----|
| Slow internet for 14B model and datasets | Start downloads first; `ollama pull` resumes if interrupted |
| CPU too slow for full grid | Drop 14B, or run 14B on dev only and say so in the report |
| Annotators disagree a lot (kappa < 0.4) | Tighten the guideline with more examples, re-annotate 20 |
| Model ignores the relation list | Enum in the schema forces valid values; count `other` usage as a metric |
| Test set leaks into tuning | Git tag before test runs; only one person runs test evaluation |

---

## 10. Definition of done

- [ ] 100 questions sampled, dev/test split saved with seed recorded
- [ ] 100 multi-sentence answers generated, ~30 planted errors recorded
- [ ] Gold claims + triples for all 100; 50 double-annotated; agreement reported
- [ ] 5 extractors working, sharing one output format
- [ ] Matcher validated against 100 human-labelled pairs
- [ ] Full ablation table on the frozen test set
- [ ] Error taxonomy with counts
- [ ] Report section, slides, README updated
- [ ] Code tagged `obj1-final` and pushed

## 11. References

1. X. Ho, A.-K. Duong Nguyen, S. Sugawara, A. Aizawa, "Constructing A Multi-hop QA Dataset for
   Comprehensive Evaluation of Reasoning Steps," COLING 2020. (2WikiMultiHopQA)
2. H. Zhang, Y. Xu, L. Perez-Beltrachini, "Fine-grained Natural Language Inference Based
   Faithfulness Evaluation for Diverse Summarisation Tasks (INFUSE)," EACL 2024.
3. P. Laban, T. Schnabel, P. N. Bennett, M. A. Hearst, "SummaC: Re-Visiting NLI-based Models for
   Inconsistency Detection in Summarization," TACL 2022.
4. Z. Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation," arXiv:2410.05779.
