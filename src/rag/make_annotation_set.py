"""Sample real LightRAG answers and write one labelling sheet per annotator (tasks D1-D3).

Answers are spread evenly over question types, with a target share of answers the automatic
filter judged incorrect or needs_review (they hold most natural errors). Each sheet row is one
answer sentence. The automatic label and sentence mapping are left out of the sheets so
annotators judge independently; labelling rules: docs/annotation_guideline.md.
"""
import argparse
import csv
import json
import random
import sys
from datetime import date
from pathlib import Path

LIGHTRAG_ROOT = Path("outputs/lightrag")
BENCH_ROOT = Path("outputs/benchmark")
ANNOT_ROOT = Path("outputs/annotation")
BAD_LABELS = {"incorrect", "needs_review"}
SHEET_COLUMNS = [
    "row_id", "answer_id", "question_type", "question", "gold_answer", "full_answer",
    "sentence_index", "sentence", "reference_triples", "verdict", "error_type",
    "evidence_edges", "notes",
]
LABEL_COLUMNS = ["verdict", "error_type", "evidence_edges", "notes"]


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Spread n over types round-robin (sorted order), never above a type's available count."""
    quota = {t: 0 for t in counts}
    left = n
    while left > 0:
        open_types = [t for t in sorted(counts) if quota[t] < counts[t]]
        if not open_types:
            break
        for t in open_types:
            if left == 0:
                break
            quota[t] += 1
            left -= 1
    return quota


def pick(group: list[dict], k: int, share: float, rng: random.Random) -> list[dict]:
    """k rows from one type, aiming for `share` incorrect/needs_review rows."""
    bad = sorted((r for r in group if r.get("label") in BAD_LABELS), key=lambda r: r["id"])
    good = sorted((r for r in group if r.get("label") not in BAD_LABELS), key=lambda r: r["id"])
    n_bad = min(len(bad), round(k * share))
    n_good = min(len(good), k - n_bad)
    n_bad = min(len(bad), k - n_good)  # top up from bad if there are too few good rows
    return rng.sample(bad, n_bad) + rng.sample(good, n_good)


def sample_answers(rows: list[dict], n: int, share: float,
                   seed: int) -> tuple[list[dict], list[str]]:
    """Choose n answers spread over question types; return (chosen, warnings)."""
    warnings = []
    if len(rows) < n:
        warnings.append(f"only {len(rows)} answers available (wanted {n}); took all")
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(str(r.get("type")), []).append(r)
    quota = allocate({t: len(v) for t, v in by_type.items()}, n)
    rng = random.Random(seed)
    chosen = []
    for t in sorted(by_type):
        chosen += pick(by_type[t], quota[t], share, rng)
    return sorted(chosen, key=lambda r: (str(r.get("type")), r["id"])), warnings


def format_triples(graph: dict) -> str:
    """Every gold edge on its own line as "[index] subject | relation | object"."""
    return "\n".join(f"[{e['index']}] {e['subject']} | {e['relation']} | {e['object']}"
                     for e in graph.get("edges", []))


def sheet_rows(row: dict) -> list[dict]:
    """One sheet row per answer sentence, label columns blank."""
    triples = format_triples(row.get("reference_graph") or {})
    return [{
        "row_id": f"{row['id']}__s{s['index']}",
        "answer_id": row["id"],
        "question_type": row.get("type"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "full_answer": row.get("answer"),
        "sentence_index": s["index"],
        "sentence": s["text"],
        "reference_triples": triples,
        **{c: "" for c in LABEL_COLUMNS},
    } for s in row.get("sentences", [])]


def write_sheet(rows: list[dict], path: Path) -> None:
    """Write one annotator sheet (UTF-8 with BOM so Excel shows accents)."""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def load_splits(bench_dir: Path) -> dict[str, str]:
    """source_id -> split from the benchmark files, if they exist."""
    splits = {}
    for split in ("dev", "test"):
        path = bench_dir / f"benchmark_{split}.jsonl"
        if path.exists():
            for r in load_jsonl(path):
                splits[r["source_id"]] = split
    return splits


def _count(rows: list[dict], key) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(key(r))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def run_self_test() -> None:
    """Run built-in cases, print PASS/FAIL, exit 1 if any fail."""
    types = ["bridge_comparison", "comparison", "compositional", "inference"]
    graph = {"edges": [
        {"index": 0, "subject": "To Sleep with Anger", "relation": "director",
         "object": "Charles Burnett"},
        {"index": 1, "subject": "Charles Burnett (director)", "relation": "date of birth",
         "object": "April 13, 1944"}]}

    def fake(i: int, label: str) -> dict:
        return {"id": f"q{i:02d}", "type": types[i % 4], "question": "Q?", "gold_answer": "A",
                "answer": "One. Two.", "label": label, "reference_graph": graph,
                "sentences": [{"index": 0, "text": "One.", "mentioned": ["x"], "edges": [0],
                               "paths": []},
                              {"index": 1, "text": "Two.", "mentioned": [], "edges": [],
                               "paths": []}]}

    ten = [fake(i, "correct" if i % 2 else "incorrect") for i in range(10)]
    four, _ = sample_answers(ten, 4, 0.6, 42)
    forty = [fake(i, "incorrect" if (i // 4) % 10 < 6 else "correct") for i in range(40)]
    twenty, _ = sample_answers(forty, 20, 0.6, 42)
    sheet = sheet_rows(ten[0])
    forbidden = {"label", "mentioned", "edges", "paths"}
    cases = [
        ("4 answers spread over types", len(four) == 4
         and sorted(r["type"] for r in four) == types),
        ("incorrect share respected", len(twenty) == 20
         and sum(r["label"] in BAD_LABELS for r in twenty) == 12),
        ("one row per sentence, columns in order, blank labels", len(sheet) == 2
         and all(list(r) == SHEET_COLUMNS for r in sheet)
         and all(r[c] == "" for r in sheet for c in LABEL_COLUMNS)),
        ("reference triples format", sheet[0]["reference_triples"]
         == "[0] To Sleep with Anger | director | Charles Burnett\n"
            "[1] Charles Burnett (director) | date of birth | April 13, 1944"),
        ("no automatic label or mapping in sheet", all(not (forbidden & set(r)) for r in sheet)
         and "incorrect" not in [str(v) for r in sheet for v in r.values()]),
        ("same seed same sample", [r["id"] for r in sample_answers(forty, 20, 0.6, 42)[0]]
         == [r["id"] for r in twenty]),
    ]
    failed = 0
    for name, ok in cases:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


def main() -> None:
    """Sample answers and write the annotation sheets, items and summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--incorrect-share", type=float, default=0.6)
    parser.add_argument("--annotators", default="A,B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    mapped_path = LIGHTRAG_ROOT / args.run_name / "mapped.jsonl"
    if not mapped_path.exists():
        sys.exit(f"ERROR: {mapped_path} not found. Run src/rag/map_sentences.py first.")
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    if not annotators:
        sys.exit("ERROR: --annotators is empty.")

    rows = [r for r in load_jsonl(mapped_path) if r.get("label") != "error"]
    chosen, warnings = sample_answers(rows, args.n, args.incorrect_share, args.seed)
    splits = load_splits(BENCH_ROOT / args.run_name)
    if not splits:
        warnings.append("no benchmark files found; split is null for every answer")

    sheet = [s for r in chosen for s in sheet_rows(r)]
    out_dir = ANNOT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in annotators:
        write_sheet(sheet, out_dir / f"sheet_{name}.csv")

    by_id = {r["id"]: r for r in chosen}
    with (out_dir / "annotation_items.jsonl").open("w", encoding="utf-8") as f:
        for s in sheet:
            src = by_id[s["answer_id"]]
            item = {**s, "answer_label": src.get("label"), "split": splits.get(src["id"]),
                    "reference_graph": src.get("reference_graph")}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for r in chosen:
        r["_split"] = splits.get(r["id"])
    sentence_rows = [{**s, "label": by_id[s["answer_id"]].get("label"),
                      "split": splits.get(s["answer_id"])} for s in sheet]
    summary = {
        "answers": len(chosen),
        "sentences": len(sheet),
        "answers_per_question_type": _count(chosen, lambda r: r.get("type")),
        "sentences_per_question_type": _count(sheet, lambda s: s["question_type"]),
        "answers_per_label": _count(chosen, lambda r: r.get("label")),
        "sentences_per_label": _count(sentence_rows, lambda s: s["label"]),
        "answers_per_split": _count(chosen, lambda r: r["_split"]),
        "sentences_per_split": _count(sentence_rows, lambda s: s["split"]),
        "annotators": annotators,
        "warnings": warnings,
        "settings": {k: v for k, v in vars(args).items() if k != "self_test"},
        "created": date.today().isoformat(),
    }
    (out_dir / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Answers: {summary['answers']}, sentences: {summary['sentences']}")
    for key in ("answers_per_question_type", "answers_per_label", "answers_per_split",
                "sentences_per_label"):
        print(f"  {key:<28}{summary[key]}")
    for w in warnings:
        print(f"WARNING: {w}")
    for name in annotators:
        print(f"Saved -> {out_dir / f'sheet_{name}.csv'}")
    print(f"Saved -> {out_dir / 'annotation_items.jsonl'}")
    print(f"Saved -> {out_dir / 'annotation_summary.json'}")


if __name__ == "__main__":
    main()
