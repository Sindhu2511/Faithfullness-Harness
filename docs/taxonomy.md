# Graph-RAG Hallucination Taxonomy

Faithfulness Evaluation Harness for Graph-Based RAG - Team 17

## 1. Purpose

This taxonomy defines the types of hallucination a Graph-RAG answer can contain, measured against
the reference knowledge graph for the question (its directed gold evidence triples; see
docs/decisions.md, D1). It is used to label every error in the benchmark and to
report which types of error the verifier gets wrong.

It is NOT the extractor error list in `docs/objective1_plan.md` (section 7.1). That list describes
mistakes the claim extractor makes while splitting text, such as merged facts or a compound
subject. This taxonomy describes mistakes the Graph-RAG answer itself makes against the graph.

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| c = (s, r, o) | One claim from the answer, written as a triple: subject s, relation r, object o |
| G | The reference knowledge graph for the question: the set of directed gold evidence triples (subject, relation, object) from 2WikiMultiHopQA. See docs/decisions.md, D1. |
| Edge exists | (s, r, o) is in G, with the same direction |
| Multi-hop path | A sequence of edges in G linking s to o through intermediate entities: (s, r1, x1), (x1, r2, x2), ..., (xk, rn, o) |

Edges are directed. (A, father, B) means "A's father is B" and is a different edge from
(B, father, A). Relation names use snake_case, for example `place_of_birth`, `spouse`, `director`.

## 3. Error types

### 3.1 Relation reversal

**Definition:** The answer uses the correct relation between the correct two entities, but in the
wrong direction.

**Formal rule:** (o, r, s) in G, but (s, r, o) not in G. Subject and object swapped.

**Triple change:** s and o swap places. r stays the same.

**Example:**

| Field | Value |
|-------|-------|
| Graph edge(s) in G | (Sofia Coppola, father, Francis Ford Coppola) |
| Correct answer sentence | Sofia Coppola's father is Francis Ford Coppola. |
| Hallucinated sentence | Francis Ford Coppola's father is Sofia Coppola. |
| Claim triple | (Francis Ford Coppola, father, Sofia Coppola) |

### 3.2 Wrong relation

**Definition:** The answer links the correct two entities, but with a relation the graph does
not hold between them.

**Formal rule:** (s, r', o) in G with r' != r, and (s, r, o) not in G. Entities right, relation
wrong.

**Triple change:** r is replaced. s and o stay the same.

**Example:**

| Field | Value |
|-------|-------|
| Graph edge(s) in G | (Lost in Translation, director, Sofia Coppola) |
| Correct answer sentence | Lost in Translation was directed by Sofia Coppola. |
| Hallucinated sentence | Sofia Coppola starred in Lost in Translation. |
| Claim triple | (Lost in Translation, cast_member, Sofia Coppola) |

### 3.3 Entity swap

**Definition:** The answer keeps the correct relation but puts in a different entity of the same
kind in place of the correct one.

**Formal rule:** (s, r, o') in G with o' != o (or (s', r, o) in G with s' != s), the swapped
entity is the same kind (person for person, place for place), and (s, r, o) not in G.

**Triple change:** o (or s) is replaced by another entity of the same kind. r stays the same.

**Example:**

| Field | Value |
|-------|-------|
| Graph edge(s) in G | (Frida Kahlo, spouse, Diego Rivera) |
| Correct answer sentence | Frida Kahlo was married to Diego Rivera. |
| Hallucinated sentence | Frida Kahlo was married to Leon Trotsky. |
| Claim triple | (Frida Kahlo, spouse, Leon Trotsky) |

### 3.4 Broken multi-hop chain

**Definition:** The answer reaches its conclusion through a chain of facts, and at least one
link in that chain is not in the graph, so the conclusion is not supported.

**Formal rule:** The claim states or depends on a path s -> x1 -> ... -> o, but at least one hop
is not in G or links to a different intermediate entity, so the path does not connect.

**Triple change:** One hop in the path is wrong or missing, so the end-to-end claim (s, path, o)
joins s to an o that G does not reach from s.

**Example:** Question: "Where was the director of Lost in Translation born?"

| Field | Value |
|-------|-------|
| Graph edge(s) in G | (Lost in Translation, director, Sofia Coppola); (Sofia Coppola, place_of_birth, New York City); (Francis Ford Coppola, place_of_birth, Detroit) |
| Correct answer sentence | The director of Lost in Translation, Sofia Coppola, was born in New York City. |
| Hallucinated sentence | The director of Lost in Translation was born in Detroit. |
| Claim triple | (Lost in Translation, director -> place_of_birth, Detroit) |

The second hop (Francis Ford Coppola, place_of_birth, Detroit) is a real edge, but the first hop
goes to Sofia Coppola, not Francis Ford Coppola, so the path does not connect.

### 3.5 Fabricated fact

**Definition:** The answer states a fact the graph holds nothing about, not even a conflicting
version.

**Formal rule:** None of rules 1-4 apply, and (s, r, o) has no support in G (no matching edge or
path, often an entity or relation absent from G).

**Triple change:** No correct triple to compare with. The relation, the object, or both are not
in G for this subject.

**Example:**

| Field | Value |
|-------|-------|
| Graph edge(s) in G | (Sofia Coppola, father, Francis Ford Coppola); (Sofia Coppola, place_of_birth, New York City) |
| Correct answer sentence | Sofia Coppola was born in New York City. |
| Hallucinated sentence | Sofia Coppola won the Nobel Prize in Literature. |
| Claim triple | (Sofia Coppola, award_received, Nobel Prize in Literature) |

G has no `award_received` edge for Sofia Coppola, so this is not an entity swap.

## 4. Decision order

Each wrong claim gets exactly ONE error type. Check the rules in this order and stop at the first
one that matches:

1. Relation reversal
2. Wrong relation
3. Entity swap
4. Broken multi-hop chain
5. Fabricated fact

**Example of overlap:**

- G: (Sofia Coppola, father, Francis Ford Coppola); (Francis Ford Coppola, father, Carmine Coppola)
- Hallucinated sentence: "Francis Ford Coppola's father is Sofia Coppola."
- Claim triple: (Francis Ford Coppola, father, Sofia Coppola)

This matches rule 1, because (Sofia Coppola, father, Francis Ford Coppola) is in G. It also
matches rule 3, because (Francis Ford Coppola, father, Carmine Coppola) is in G and Sofia Coppola
is a person in place of a person. Rule 1 comes first, so the label is **relation reversal**.

## 5. Verdict mapping

| Verdict | When it applies |
|---------|-----------------|
| supported | (s, r, o) is an edge in G, or s and o are joined by a valid path in G |
| contradicted | Types 1-4: G holds a conflicting fact |
| unsupported | Type 5: G holds nothing on the claim |

Note: this mapping is a draft to confirm with the guide.

## 6. Comparison with existing work

### 6.1 Do existing checkers name these error types?

| Error type | GraphEval (Sansford 2024) | RefChecker (Hu 2024) | GGA (Li 2025) |
|---|---|---|---|
| Relation reversal | Not named. Caught only as a generic inconsistent triple | Not named. Caught only as generic Contradiction | Not named. Only a whole-answer hallucinated/truthful label |
| Wrong relation | Not named. Generic inconsistent triple | Not named. Generic Contradiction | Not named. Whole-answer label only |
| Entity swap | Not named. Generic inconsistent triple | Not named. Generic Contradiction | Not named. Whole-answer label only |
| Broken multi-hop chain | Not named. No path checking | Not named. No path checking | Not addressed. Single-hop questions only (MetaQA-1hop); multi-hop left to future work |
| Fabricated fact | Not named. Generic inconsistent triple | Closest is the Neutral label (claim cannot be verified from the reference) | Not named. Whole-answer label only |

### 6.2 How each checker works

| Aspect | GraphEval | RefChecker | GGA | This project |
|---|---|---|---|---|
| Evidence form | Plain text context | Plain text reference | Linearised KG subgraph in the prompt | Reference knowledge graph (gold evidence triples), with LightRAG's retrieved context as supporting evidence |
| Checking unit | Triples built from the LLM output | Claim triples | Whole answer | Claim triples and multi-hop paths |
| Check method | NLI model per triple | LLM or NLI checker per triple | Attention and hidden-state signals plus surface features, XGBoost classifier | Symbolic graph checks first, NLI fallback |
| Labels | Binary (consistent / inconsistent) | 3-way (Entailment / Contradiction / Neutral) | Binary (hallucinated / truthful) | 3-way verdict plus one of 5 error types plus evidence path |
| Needs model internals | No | No | Yes | No |
| Evaluation data | SummEval, QAGS-C, QAGS-X (summarisation) | Own benchmark, 11k human-labelled triples from 2.1k responses | MetaQA-1hop | Injected-error benchmark plus about 100 human-labelled answers |

### 6.3 Conclusion

None of the three checkers names any of the 5 error types. GraphEval and RefChecker check claims
against text, not graph structure. GGA uses model internals, not graph structure, and covers
single-hop questions only. The novel part of this project is typing each error and checking graph
structure (edge, direction, path) before falling back to NLI.

Still to check before claiming novelty: FactGraph (Ribeiro et al. 2022) and "LLMs hallucinate
graphs too: a structural perspective" (Le Merrer and Trédan 2024).

## 7. Sign-off

- Reviewed by: Ms. Suparna Das, Assistant Professor, Department of CSE
- Date:
- Changes requested:
- Approved (yes/no):
