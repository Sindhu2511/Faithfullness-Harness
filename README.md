\# Faithfulness Evaluation Harness for Graph-Based RAG



A system that checks whether answers produced by a Graph-RAG system are actually supported by the knowledge graph they were generated from. Each answer is split into individual factual claims, every claim is matched to graph evidence, and an NLI model labels it as supported, contradicted, or unsupported. The labels are combined into faithfulness and hallucination scores with a per-claim explanation.



\*\*Team 17, Department of Information Technology, BVRIT Hyderabad College of Engineering for Women\*\*

K. Sree Sindhu (23WH1A1251), V. Malleswari (23WH1A1229), M. Jahnavi (23WH1A1242), S. Sahasra (23WH1A1203)

Guide: Ms. Suparna Das, Assistant Professor, Department of CSE



\## Pipeline



1\. Query input

2\. Graph-based retrieval (entities, relationship edges, community summaries)

3\. Response generation by an LLM

4\. Claim extraction

5\. Evidence alignment

6\. NLI classification (supported / contradicted / unsupported)

7\. Faithfulness and hallucination scoring

8\. Evaluation report



\## Objectives



| # | Objective | Status |

|---|-----------|--------|

| 1 | Fine-grained claim extraction: split responses into atomic claims with (subject, relation, object) triples | Done |

| 2 | Graph-based evidence verification: link claims to entities, edges, and community summaries in Neo4j | Planned |

| 3 | Entity-relationship consistency checking with NLI | Planned |

| 4 | Hallucination detection and faithfulness scoring | Planned |



\## Project structure



```

faithfulness-harness/

├── data/

│   └── test\_responses.json    # test responses with hand-written gold claims

├── src/

│   ├── baseline.py            # spaCy baseline: one sentence = one claim

│   ├── extractor.py           # LLM extractor: atomic claims + triples via Ollama

│   └── evaluate.py            # scores extractors against gold claims

├── outputs/                   # generated claims and evaluation report

├── pyproject.toml

└── uv.lock

```



\## Setup



Requirements: Python 3.12, \[uv](https://docs.astral.sh/uv/), \[Ollama](https://ollama.com/).



```powershell

git clone https://github.com/Sindhu2511/Faithfullness-Harness.git

cd Faithfullness-Harness

uv sync

ollama pull qwen2.5:7b

```



\## Usage



```powershell

uv run python src/baseline.py     # spaCy baseline -> outputs/baseline\_claims.json

uv run python src/extractor.py    # LLM extractor  -> outputs/llm\_claims.json

uv run python src/evaluate.py     # scores both    -> outputs/eval\_report.json

```



\## Objective 1 results (development set: 6 responses, 18 gold claims)



| Metric | spaCy baseline | LLM extractor |

|--------|----------------|---------------|

| Precision | 0.40 | 0.94 |

| Recall | 0.22 | 0.83 |

| F1 | 0.29 | 0.88 |

| Planted errors kept | 0/2 | 2/2 |

| Pronoun leaks | 4 | 0 |



\*\*Limitations:\*\*

\- The set is small, and the prompt was tuned on it, so these scores are optimistic. Held-out evaluation is pending.

\- Count facts ("won two Nobel Prizes") and sentences linking three entities are sometimes not split.

\- Claim matching uses character similarity, which can credit claims with unresolved pronouns. It will be replaced with NLI-based matching in Objective 3.



\## Tech stack



Python 3.12, uv, spaCy, LangChain, Ollama (qwen2.5:7b), Pydantic. Planned: Neo4j, Hugging Face Transformers, Sentence Transformers, Streamlit.



\## References



1\. H. Zhang, Y. Xu, L. Perez-Beltrachini, "Fine-grained Natural Language Inference Based Faithfulness Evaluation for Diverse Summarisation Tasks (INFUSE)," EACL 2024.

2\. Z. Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation," arXiv:2410.05779, 2024.

3\. P. Laban, T. Schnabel, P. N. Bennett, M. A. Hearst, "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization," TACL 2022.

