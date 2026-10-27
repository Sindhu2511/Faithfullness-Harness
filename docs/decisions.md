# Design Decisions

## D1. Source of edge direction: gold evidence triples

- Date: 2026-10-27
- Status: Accepted (to confirm with guide)

### Context

LightRAG 1.5.7 stores its knowledge graph undirected. NetworkXStorage in
`lightrag/kg/networkx_impl.py` uses `nx.Graph()`, and retrieved relationship pairs are sorted
alphabetically (`tuple(sorted(e))` in `_find_most_related_edges_from_entities`,
`lightrag/operate.py`). So (A, r, B) and (B, r, A) look identical in LightRAG's retrieved
subgraph. Relation reversal (type 1) and entity swap (type 3) in `docs/taxonomy.md` need a
directed graph.

### Options considered

A. Use the dataset's gold evidence triples (2WikiMultiHopQA `evidences`, directed
   Wikidata-style triples) as the reference graph.

B. Infer direction from LightRAG relation descriptions with an LLM.

C. Both A and B.

### Decision

Option A. The reference graph G for each question is its gold `evidences` triples. LightRAG is
used only to generate answers; its retrieved entities, relations and chunks are kept as
supporting evidence and for the NLI fallback.

### Consequences

- Benchmark labels are reliable: injected errors change gold triples whose direction is known.
- The verifier checks claims against the gold reference graph, not only against what LightRAG
  retrieved. Objective 2 wording changes from "retrieved subgraph" to "reference knowledge graph
  (gold evidences), with LightRAG's retrieved context as supporting evidence".
- Entity names in gold evidences are not always consistent (example: "Charles Burnett" and
  "Charles Burnett (director)" are the same person). Entity normalisation is required before
  path checks.
- Finding for the report: a checker that relies on LightRAG's own graph cannot detect relation
  reversal, because that graph has no direction.
