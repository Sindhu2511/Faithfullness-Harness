"""Map each LightRAG answer sentence to the gold edges or multi-hop paths it states (task B5).

For every answer, the question's directed reference graph is built from its gold evidences
(docs/decisions.md, D1). A sentence that mentions both ends of a gold edge states that edge;
a sentence that mentions two nodes joined only by a path (e.g. film and birth date, but not the
director) states that path. Phase C uses this mapping to choose which sentence to edit.
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from entities import build_reference_graph, find_paths, mentions, normalise_entity

OUT_ROOT = Path("outputs/lightrag")
# Words ending in "." that do not end a sentence (compared lowercase, without the dot).
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc", "ltd", "co", "corp", "vs",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "gen", "col", "lt", "sgt", "capt", "rev", "hon", "mt", "ft", "approx", "etc", "e.g", "i.e",
}
BOUNDARY_RE = re.compile(r"[.!?]\s+")


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? + whitespace + uppercase letter or digit, but not after initials/abbreviations."""
    text = (text or "").strip()
    sentences, start = [], 0
    for m in BOUNDARY_RE.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if not (nxt.isupper() or nxt.isdigit()):
            continue
        if text[m.start()] == ".":
            prev = re.search(r"(\S+)$", text[start:m.start()])
            word = prev.group(1) if prev else ""
            # "J." / "R." initials, "U.S." style dotted words, and known abbreviations.
            if (len(word) == 1 and word.isalpha()) or "." in word \
                    or word.lower() in ABBREVIATIONS:
                continue
        sentences.append(text[start:m.start() + 1].strip())
        start = m.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def map_answer(answer: str, graph: dict, max_hops: int) -> dict:
    """Per-sentence mentioned nodes, direct edges and implied paths, plus unmapped/unused."""
    linked = {frozenset((e["subject_key"], e["object_key"])) for e in graph["edges"]}
    sentences, covered = [], set()
    for i, text in enumerate(split_sentences(answer)):
        mentioned = [n["key"] for n in graph["nodes"] if mentions(text, n)]
        mset = set(mentioned)
        edges = [e["index"] for e in graph["edges"]
                 if e["subject_key"] in mset and e["object_key"] in mset]
        paths: list[list[int]] = []
        for a in mentioned:
            for b in mentioned:
                if a == b or frozenset((a, b)) in linked:
                    continue
                for p in find_paths(graph, a, b, max_hops):
                    if p not in paths:
                        paths.append(p)
        covered.update(edges)
        for p in paths:
            covered.update(p)
        sentences.append({"index": i, "text": text, "mentioned": mentioned,
                          "edges": edges, "paths": paths})
    return {
        "sentences": sentences,
        "unmapped_sentences": [s["index"] for s in sentences if not s["edges"] and not s["paths"]],
        "unused_edges": [e["index"] for e in graph["edges"] if e["index"] not in covered],
    }


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_self_test() -> None:
    """Run built-in cases, print PASS/FAIL, exit 1 if any fail."""
    ev = [["To Sleep with Anger", "director", "Charles Burnett"],
          ["Charles Burnett (director)", "date of birth", "April 13, 1944"]]
    g = build_reference_graph(ev)
    film = normalise_entity("To Sleep with Anger")
    dob = normalise_entity("April 13, 1944")
    direct = map_answer("To Sleep with Anger was directed by Charles Burnett.", g, 2)
    implied = map_answer("The director of To Sleep with Anger was born on April 13, 1944.", g, 2)
    unmapped = map_answer("Paris is a large city. To Sleep with Anger was directed by "
                          "Charles Burnett.", g, 2)
    cases = [
        ("alias key", normalise_entity("Charles Burnett (director)")
         == normalise_entity("Charles Burnett")),
        ("accents", normalise_entity("Zdeněk Troška") == "zdenek troska"),
        ("date any order", mentions("He was born on 13 April 1944.",
                                    {"key": dob, "names": ["April 13, 1944"]})),
        ("one alias merge", len(g["alias_merges"]) == 1),
        ("path film->date", find_paths(g, film, dob, 2) == [[0, 1]]),
        ("direct edge", direct["sentences"][0]["edges"] == [0]
         and direct["sentences"][0]["paths"] == []),
        ("implied path", implied["sentences"][0]["edges"] == []
         and implied["sentences"][0]["paths"] == [[0, 1]]),
        ("unmapped sentence", unmapped["unmapped_sentences"] == [0]),
        ("split keeps initials", len(split_sentences(
            "J. R. R. Tolkien wrote it. He was born on Jan. 3, 1892. 3 films followed.")) == 3),
    ]
    failed = 0
    for name, ok in cases:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


def main() -> None:
    """Map every answer of a run and write mapped.jsonl and map_summary.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--questions", type=Path, default=Path("data/questions/questions.jsonl"))
    parser.add_argument("--source", default="judged", choices=["judged", "clean"])
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    out_dir = OUT_ROOT / args.run_name
    source_path = out_dir / f"{args.source}.jsonl"
    if not source_path.exists():
        sys.exit(f"ERROR: {source_path} not found. Run src/rag/filter_correct.py first.")
    if not args.questions.exists():
        sys.exit(f"ERROR: {args.questions} not found. Run src/data/prepare_questions.py first.")
    evidences = {q["id"]: q["evidences"] for q in load_jsonl(args.questions)}

    records, skipped_error, missing = [], 0, []
    for row in load_jsonl(source_path):
        if row.get("label") == "error":
            skipped_error += 1
            continue
        if row["id"] not in evidences:
            missing.append(row["id"])
            continue
        graph = build_reference_graph(evidences[row["id"]])
        mapping = map_answer(row.get("answer") or "", graph, args.max_hops)
        records.append({
            "id": row["id"], "type": row.get("type"), "question": row.get("question"),
            "gold_answer": row.get("gold_answer"), "answer": row.get("answer"),
            "label": row.get("label"), "reference_graph": graph, **mapping,
        })

    with (out_dir / "mapped.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_sent = sum(len(r["sentences"]) for r in records)
    n_unmapped = sum(len(r["unmapped_sentences"]) for r in records)
    per_label: dict[str, int] = {}
    for r in records:
        per_label[str(r["label"])] = per_label.get(str(r["label"]), 0) + 1
    n = len(records)
    summary = {
        "rows_mapped": n,
        "rows_skipped": {"label_error": skipped_error, "missing_question": len(missing),
                         "missing_question_ids": missing},
        "total_sentences": n_sent,
        "sentences_with_edge_or_path": n_sent - n_unmapped,
        "unmapped_sentences": n_unmapped,
        "avg_gold_edges_per_answer": round(
            sum(len(r["reference_graph"]["edges"]) for r in records) / n, 2) if n else None,
        "avg_covered_edges_per_answer": round(
            sum(len(r["reference_graph"]["edges"]) - len(r["unused_edges"]) for r in records) / n,
            2) if n else None,
        "total_alias_merges": sum(len(r["reference_graph"]["alias_merges"]) for r in records),
        "rows_per_label": dict(sorted(per_label.items())),
        "max_hops": args.max_hops,
        "source": args.source,
        "created": date.today().isoformat(),
    }
    (out_dir / "map_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Source: {source_path}")
    for key, value in summary.items():
        if key != "created":
            print(f"  {key:<30}{value}")
    print(f"Saved -> {out_dir / 'mapped.jsonl'}")
    print(f"Saved -> {out_dir / 'map_summary.json'}")


if __name__ == "__main__":
    main()
