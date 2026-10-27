"""Run LightRAG over 2WikiMultiHopQA questions and save answers plus retrieved subgraphs (task B3).

Builds one LightRAG index per run from the questions' context paragraphs (Ollama for LLM and
embeddings), then asks each question once with aquery_llm, which returns the answer and the
structured retrieval data (entities, relationships, chunks) from the SAME query call.
Both indexing and answering are resumable: re-running skips finished documents and questions.

Note: LightRAG 1.5.7 stores its graph in an undirected networkx.Graph and returns relationship
endpoints in sorted (alphabetical) order, so src/tgt in the output do NOT encode direction.
"""
import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import date
from functools import partial
from pathlib import Path

from lightrag import LightRAG, QueryParam, __version__ as LIGHTRAG_VERSION
from lightrag.base import DocStatus
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

# Read from lightrag/kg/networkx_impl.py (NetworkXStorage uses nx.Graph) and
# lightrag/operate.py (_find_most_related_edges_from_entities sorts each edge pair).
GRAPH_IS_DIRECTED = False

# Modes from the installed QueryParam.mode Literal (lightrag/base.py).
MODES = re.findall(r"['\"](\w+)['\"]", str(QueryParam.__annotations__["mode"]))

ANSWER_PROMPT = ("Answer only from the retrieved context, in 1-3 sentences. "
                 "Do not add facts that are not in the context.")
INSERT_BATCH = 10


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--questions", type=Path, default=Path("data/questions/questions.jsonl"))
    p.add_argument("--limit", type=int, default=20, help="pilot size; 0 means all")
    p.add_argument("--run-name", default="pilot")
    p.add_argument("--llm-model", default="qwen2.5:3b")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--mode", default="hybrid", choices=MODES)
    p.add_argument("--max-async", type=int, default=1)
    p.add_argument("--llm-timeout", type=int, default=900,
                   help="seconds per LLM call (CPU is slow)")
    return p.parse_args()


def load_questions(path: Path, limit: int) -> list[dict]:
    """Load the first `limit` questions in file order (0 = all)."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run src/data/prepare_questions.py first.")
    with path.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return questions[:limit] if limit > 0 else questions


def doc_id_for(title: str) -> str:
    """Stable document id derived from the paragraph title."""
    return "doc-" + hashlib.md5(title.encode("utf-8")).hexdigest()


def build_corpus(questions: list[dict]) -> dict[str, dict]:
    """One document per context paragraph, deduplicated by title. Returns {doc_id: doc}."""
    corpus: dict[str, dict] = {}
    for q in questions:
        for para in q["context"]:
            doc_id = doc_id_for(para["title"])
            if doc_id not in corpus:
                corpus[doc_id] = {
                    "title": para["title"],
                    "text": para["title"] + "\n" + " ".join(para["sentences"]),
                }
    return corpus


def build_rag(args: argparse.Namespace, working_dir: Path) -> LightRAG:
    """Construct LightRAG with Ollama for both the LLM and the embeddings."""
    embedding = EmbeddingFunc(
        embedding_dim=args.embed_dim,
        max_token_size=args.num_ctx,
        model_name=args.embed_model,
        # .func is the unwrapped function; passing ollama_embed itself would double-wrap.
        func=partial(ollama_embed.func, embed_model=args.embed_model,
                     host=args.ollama_host, timeout=args.llm_timeout),
    )
    return LightRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=args.llm_model,
        llm_model_kwargs={
            "host": args.ollama_host,
            "options": {"num_ctx": args.num_ctx},
            "timeout": args.llm_timeout,  # ollama client timeout
        },
        llm_model_max_async=args.max_async,
        default_llm_timeout=args.llm_timeout,  # LightRAG worker timeout
        embedding_func=embedding,
        embedding_func_max_async=args.max_async,
        default_embedding_timeout=args.llm_timeout,
        max_parallel_insert=args.max_async,
    )


def _status_of(record) -> str | None:
    """Status value from a doc_status record (dict or DocProcessingStatus)."""
    if record is None:
        return None
    status = record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
    return getattr(status, "value", status)


async def index_corpus(rag: LightRAG, corpus: dict[str, dict]) -> int:
    """Insert documents not yet in LightRAG, retry unfinished ones; return processed count."""
    ids = list(corpus)
    statuses = {i: _status_of(await rag.doc_status.get_by_id(i)) for i in ids}
    new_ids = [i for i in ids if statuses[i] is None]
    unfinished = [i for i in ids if statuses[i] not in (None, DocStatus.PROCESSED.value)]
    print(f"Corpus: {len(ids)} documents, {len(ids) - len(new_ids) - len(unfinished)} already "
          f"indexed, {len(unfinished)} unfinished, {len(new_ids)} new")

    for start in range(0, len(new_ids), INSERT_BATCH):
        batch = new_ids[start:start + INSERT_BATCH]
        t0 = time.time()
        await rag.ainsert(
            [corpus[i]["text"] for i in batch],
            ids=batch,
            # file_path is stored as a basename, so path separators in titles are replaced.
            file_paths=[re.sub(r"[\\/]", "_", corpus[i]["title"]) for i in batch],
        )
        print(f"  indexed {start + len(batch)}/{len(new_ids)} new documents "
              f"({time.time() - t0:.1f}s)")
    if unfinished and not new_ids:
        # ainsert already reprocesses pending/failed docs; only needed when nothing new was added.
        await rag.apipeline_process_enqueue_documents()

    done = [i for i in ids
            if _status_of(await rag.doc_status.get_by_id(i)) == DocStatus.PROCESSED.value]
    if len(done) < len(ids):
        print(f"WARNING: {len(ids) - len(done)} documents are not processed; "
              "re-run to retry them.")
    return len(done)


def load_done_ids(path: Path) -> set[str]:
    """Question ids already present in answers.jsonl."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


async def answer_question(rag: LightRAG, q: dict, args: argparse.Namespace) -> dict:
    """Ask one question; return the output record (answer + retrieved subgraph)."""
    record = {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "gold_answer": q["gold_answer"], "answer": None, "mode": args.mode,
        "llm_model": args.llm_model, "embed_model": args.embed_model,
        "retrieved_entities": [], "retrieved_relations": [], "retrieved_chunk_ids": [],
        "seconds": None, "error": None,
    }
    param = QueryParam(
        mode=args.mode,
        response_type="1-3 sentences",
        user_prompt=ANSWER_PROMPT,
        disable_user_prompt_prefix=True,
        stream=False,
    )
    start = time.time()
    try:
        # One call: retrieval data and the LLM answer come from the same query.
        result = await rag.aquery_llm(q["question"], param=param)
        data = result.get("data") or {}
        record["answer"] = (result.get("llm_response") or {}).get("content")
        record["retrieved_entities"] = [
            {"name": e.get("entity_name"), "type": e.get("entity_type"),
             "description": e.get("description")}
            for e in data.get("entities", [])
        ]
        record["retrieved_relations"] = [
            {"source": r.get("src_id"), "target": r.get("tgt_id"),
             "keywords": r.get("keywords"), "description": r.get("description"),
             "weight": r.get("weight")}
            for r in data.get("relationships", [])
        ]
        record["retrieved_chunk_ids"] = [c.get("chunk_id") for c in data.get("chunks", [])]
        if result.get("status") != "success":
            record["error"] = result.get("message") or "query failed"
    except Exception as e:  # keep going on any single-question failure
        record["error"] = f"{type(e).__name__}: {e}"
    record["seconds"] = round(time.time() - start, 1)
    return record


async def run(args: argparse.Namespace) -> None:
    """Index the corpus, answer every pending question, write outputs and summary."""
    questions = load_questions(args.questions, args.limit)
    if not questions:
        sys.exit(f"ERROR: no questions loaded from {args.questions}.")
    corpus = build_corpus(questions)

    working_dir = Path("data/lightrag") / args.run_name
    out_dir = Path("outputs/lightrag") / args.run_name
    working_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"

    rag = build_rag(args, working_dir)
    await rag.initialize_storages()  # also initializes pipeline status in 1.5.7
    try:
        n_indexed = await index_corpus(rag, corpus)

        done_ids = load_done_ids(answers_path)
        todo = [q for q in questions if q["id"] not in done_ids]
        print(f"Questions: {len(questions)} loaded, {len(done_ids & {q['id'] for q in questions})} "
              f"already answered, {len(todo)} to run")
        with answers_path.open("a", encoding="utf-8") as f:
            for k, q in enumerate(todo, start=1):
                rec = await answer_question(rag, q, args)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                status = "ok" if rec["error"] is None else f"ERROR: {rec['error']}"
                print(f"{k}/{len(todo)} {status} {rec['seconds']}s")
    finally:
        await rag.finalize_storages()

    ids = {q["id"] for q in questions}
    with answers_path.open(encoding="utf-8") as f:
        records = [r for r in (json.loads(line) for line in f if line.strip()) if r["id"] in ids]
    total_seconds = sum(r["seconds"] or 0 for r in records)
    summary = {
        "questions": len(questions),
        "documents": len(corpus),
        "documents_indexed": n_indexed,
        "answered": sum(1 for r in records if r["error"] is None),
        "errors": sum(1 for r in records if r["error"] is not None),
        "total_seconds": round(total_seconds, 1),
        "avg_seconds": round(total_seconds / len(records), 1) if records else None,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "lightrag_version": LIGHTRAG_VERSION,
        "graph_is_directed": GRAPH_IS_DIRECTED,
        "created": date.today().isoformat(),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary")
    for key in ("questions", "documents", "documents_indexed", "answered", "errors",
                "total_seconds", "avg_seconds", "lightrag_version", "graph_is_directed"):
        print(f"  {key:<20}{summary[key]}")
    print(f"Saved -> {answers_path}")
    print(f"Saved -> {out_dir / 'run_summary.json'}")


def main() -> None:
    """Entry point."""
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
