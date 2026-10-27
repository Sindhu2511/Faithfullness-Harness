# Annotation Guideline

Faithfulness Evaluation Harness for Graph-Based RAG - Team 17

This guide is for labelling the sheets in `outputs/annotation/<run-name>/sheet_<name>.csv`.
You do not need to read any code.

## 1. Task

Each row of your sheet is one sentence from an answer that a Graph-RAG system gave to a
question. Read the question, the reference triples (the gold facts for that question) and the
sentence. Then label the sentence: is it supported by the reference triples, contradicted by
them, or not covered by them, and if it is wrong, what kind of error is it?

The full answer is in the `full_answer` column for context, but you label only the sentence in
the `sentence` column.

## 2. verdict

Fill the `verdict` column with exactly one of these values (matches `docs/taxonomy.md`,
section 5):

| verdict | Meaning |
|---------|---------|
| supported | The sentence's fact is a reference triple, or follows from a chain of reference triples. |
| contradicted | The reference triples hold a conflicting fact (error types relation_reversal, wrong_relation, entity_swap, broken_chain). |
| unsupported | The reference triples say nothing about this fact (error type fabricated_fact). |
| no_claim | The sentence makes no factual claim, e.g. "I could not find that information." |

## 3. error_type

Fill the `error_type` column with exactly one of these values (definitions from
`docs/taxonomy.md`, section 3):

| error_type | Definition |
|------------|------------|
| none | Use when the verdict is supported or no_claim. |
| relation_reversal | The correct relation between the correct two entities, but in the wrong direction. |
| wrong_relation | The correct two entities, but linked by a relation the reference triples do not hold between them. |
| entity_swap | The correct relation, but a different entity of the same kind (person for person, place for place, date for date) in place of the correct one. |
| broken_chain | The sentence reaches its conclusion through a chain of facts, and at least one link of that chain is not in the reference triples, so the conclusion is not supported. |
| fabricated_fact | A fact the reference triples hold nothing about, not even a conflicting version. |

## 4. When two error types seem to fit

A wrong sentence gets exactly one error type. Check them in this order and take the first one
that fits:

1. relation_reversal
2. wrong_relation
3. entity_swap
4. broken_chain
5. fabricated_fact

Example: the triples are `Sofia Coppola | father | Francis Ford Coppola` and
`Francis Ford Coppola | father | Carmine Coppola`. The sentence "Francis Ford Coppola's father is
Sofia Coppola." looks like a relation_reversal (the first triple, backwards) and also like an
entity_swap (Sofia instead of Carmine). relation_reversal comes first, so that is the label.

## 5. evidence_edges

Fill `evidence_edges` with the numbers (shown in square brackets in `reference_triples`) of the
triples that support or contradict the sentence, separated by commas, for example `0,2`.
Leave it blank when the verdict is unsupported or no_claim.

## 6. Judgement rules

- Judge only against the reference triples, never against your own knowledge. A sentence that
  is true in the real world but not in the triples is unsupported; a sentence that contradicts
  the triples is contradicted even if you believe it is true.
- Resolve pronouns yourself: "He was born in 1944" refers to the person named before it in the
  answer.
- The same date in a different format counts as the same date: "13 April 1944" is the same as
  "April 13, 1944".
- Inverse relations count as the same fact: "A's father is B" is the same as "A is B's child".
- A sentence with several facts: if ANY fact is contradicted, label the sentence contradicted
  with that fact's error type; otherwise, if any fact is unsupported, label it unsupported.
  Mention the other facts in `notes`.
- Do not talk to the other annotator about the labels until both sheets are finished.

## 7. Worked examples

| Reference triples | Sentence | verdict | error_type | evidence_edges |
|-------------------|----------|---------|------------|----------------|
| [0] Sofia Coppola \| father \| Francis Ford Coppola | Sofia Coppola's father is Francis Ford Coppola. | supported | none | 0 |
| [0] Sofia Coppola \| father \| Francis Ford Coppola | Francis Ford Coppola's father is Sofia Coppola. | contradicted | relation_reversal | 0 |
| [0] Lost in Translation \| director \| Sofia Coppola | Sofia Coppola starred in Lost in Translation. | contradicted | wrong_relation | 0 |
| [0] Frida Kahlo \| spouse \| Diego Rivera | Frida Kahlo was married to Leon Trotsky. | contradicted | entity_swap | 0 |
| [0] Lost in Translation \| director \| Sofia Coppola; [1] Sofia Coppola \| place of birth \| New York City; [2] Francis Ford Coppola \| place of birth \| Detroit | The director of Lost in Translation was born in Detroit. | contradicted | broken_chain | 0,1 |
| [0] Sofia Coppola \| father \| Francis Ford Coppola; [1] Sofia Coppola \| place of birth \| New York City | Sofia Coppola won the Nobel Prize in Literature. | unsupported | fabricated_fact | |
| [0] Sofia Coppola \| place of birth \| New York City | I could not find that information. | no_claim | none | |

Note on the broken_chain row: the chain from the film to its director's birthplace goes through
triples 0 and 1 and ends in New York City, which conflicts with the sentence; triple 2 is a real
fact about a different person, so it is not the evidence.

## 8. Saving your sheet

- Open the CSV in Excel, fill the four label columns (`verdict`, `error_type`,
  `evidence_edges`, `notes`), and save it as **CSV UTF-8 (Comma delimited) (*.csv)**.
- Do not rename, reorder or delete columns, and do not change any column except the four label
  columns.
- Keep the file name as it is (`sheet_<your name>.csv`).
