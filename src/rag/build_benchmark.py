"""Balance injected and clean samples, split by question, and write the fixed benchmark schema.

Covers task C5 (balancing per error type plus clean samples), C4 (spot-check sheet) and
E1-E2 (fixed schema, dev/test split by question). Field meanings: docs/benchmark_schema.md.
"""
import argparse
import csv
import json
import random
import sys
from datetime import date
from pathlib import Path

from inject_errors import INJECTORS

LIGHTRAG_ROOT = Path("outputs/lightrag")
BENCH_ROOT = Path("outputs/benchmark")
ERROR_TYPES = list(INJECTORS)
SCHEMA_FIELDS = [
    "sample_id", "split", "source_id", "question_type", "question", "gold_answer", "answer",
    "error_type", "expected_verdict", "sentence_index", "original_sentence", "new_sentence",
    "original_triple", "new_triple", "path_edges", "wrong_intermediate", "reference_graph",
    "provenance",
]
SPOTCHECK_COLUMNS = ["sample_id", "error_type", "original_sentence", "new_sentence",
                     "original_triple", "new_triple", "reads_naturally", "label_correct", "notes"]


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_clean(row: dict) -> dict:
    """A clean (unmodified) sample from a correct mapped row."""
    return {
        "sample_id": f"{row['id']}__none",
        "source_id": row["id"],
        "question_type": row.get("type"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "error_type": "none",
        "expected_verdict": "supported",
        "injected_answer": row.get("answer"),
        "reference_graph": row.get("reference_graph"),
    }


def balance(samples: list[dict], types: list[str], per_type: int,
            seed: int) -> tuple[dict[str, list[dict]], int, list[str]]:
    """Sample each type down to the target; return (selected per type, target, warnings)."""
    by_type: dict[str, list[dict]] = {t: [] for t in types}
    for s in samples:
        by_type.setdefault(s["error_type"], []).append(s)
    warnings = [f"error type '{t}' has 0 samples; left out" for t, v in by_type.items() if not v]
    present = {t: sorted(v, key=lambda s: s["sample_id"]) for t, v in by_type.items() if v}
    target = per_type or (min(len(v) for v in present.values()) if present else 0)
    rng = random.Random(seed)
    selected = {}
    for t in sorted(present):
        group = present[t]
        if len(group) < target:
            warnings.append(f"error type '{t}' has only {len(group)} samples (target {target}); "
                            "kept all")
            selected[t] = group
        else:
            selected[t] = rng.sample(group, target)
    return selected, target, warnings


def sample_clean(clean: list[dict], count: int, seed: int) -> tuple[list[dict], list[str]]:
    """Sample clean rows down to count; warn if there are fewer."""
    clean = sorted(clean, key=lambda s: s["sample_id"])
    if len(clean) < count:
        return clean, [f"only {len(clean)} clean samples (wanted {count}); kept all"]
    return random.Random(seed).sample(clean, count), []


def split_by_question(source_ids: list[str], dev_fraction: float, seed: int) -> dict[str, str]:
    """Shuffle distinct source ids with the seed; first dev_fraction go to dev, rest to test."""
    ids = sorted(set(source_ids))
    random.Random(seed).shuffle(ids)
    n_dev = round(len(ids) * dev_fraction)
    return {sid: ("dev" if i < n_dev else "test") for i, sid in enumerate(ids)}


def to_record(sample: dict, split: str, provenance: dict) -> dict:
    """One sample in the fixed schema, fields in order, null for anything missing."""
    values = {**sample, "split": split, "answer": sample.get("injected_answer"),
              "provenance": provenance}
    return {f: values.get(f) for f in SCHEMA_FIELDS}


def build_records(samples: list[dict], dev_fraction: float, seed: int,
                  provenance_for: dict[str, dict]) -> list[dict]:
    """Assign splits by question and convert every sample to the schema."""
    splits = split_by_question([s["source_id"] for s in samples], dev_fraction, seed)
    return [to_record(s, splits[s["source_id"]], provenance_for.get(s["source_id"], {}))
            for s in samples]


def write_spotcheck(records: list[dict], per_type: int, seed: int, path: Path) -> None:
    """Up to per_type random injected samples per error type, for manual review."""
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPOTCHECK_COLUMNS)
        w.writeheader()
        for t in ERROR_TYPES:
            group = sorted((r for r in records if r["error_type"] == t),
                           key=lambda r: r["sample_id"])
            for r in rng.sample(group, min(per_type, len(group))):
                w.writerow({
                    "sample_id": r["sample_id"], "error_type": t,
                    "original_sentence": r["original_sentence"] or "",
                    "new_sentence": r["new_sentence"] or "",
                    "original_triple": json.dumps(r["original_triple"], ensure_ascii=False),
                    "new_triple": json.dumps(r["new_triple"], ensure_ascii=False),
                    "reads_naturally": "", "label_correct": "", "notes": "",
                })


def summarise(records: list[dict]) -> dict:
    """Counts per error type, clean, and question type, per split, plus the leakage check."""
    out: dict = {"per_error_type": {}, "clean": {}, "per_question_type": {}}
    for split in ("dev", "test"):
        rows = [r for r in records if r["split"] == split]
        per_err: dict[str, int] = {}
        per_q: dict[str, int] = {}
        for r in rows:
            if r["error_type"] != "none":
                per_err[r["error_type"]] = per_err.get(r["error_type"], 0) + 1
            per_q[str(r["question_type"])] = per_q.get(str(r["question_type"]), 0) + 1
        out["per_error_type"][split] = dict(sorted(per_err.items()))
        out["clean"][split] = sum(1 for r in rows if r["error_type"] == "none")
        out["per_question_type"][split] = dict(sorted(per_q.items()))
    dev = {r["source_id"] for r in records if r["split"] == "dev"}
    test = {r["source_id"] for r in records if r["split"] == "test"}
    out["leakage_check"] = {"source_ids_dev": len(dev), "source_ids_test": len(test),
                            "overlap": sorted(dev & test), "passed": not (dev & test)}
    return out


def run_self_test() -> None:
    """Run built-in cases, print PASS/FAIL, exit 1 if any fail."""
    def fake(t: str, i: int, source: str | None = None) -> dict:
        return {"sample_id": f"q{i}__{t}", "source_id": source or f"q{i}", "error_type": t,
                "expected_verdict": "contradicted", "injected_answer": "x"}

    samples = ([fake("a", i) for i in range(5)] + [fake("b", i) for i in range(3)]
               + [fake("c", i) for i in range(8)])
    selected, target, _ = balance(samples, ["a", "b", "c"], 0, 42)
    _, _, warn = balance(samples, ["a", "b", "c", "d"], 0, 42)
    clean = make_clean({"id": "q9", "type": "inference", "question": "Q?", "gold_answer": "A",
                        "answer": "A.", "reference_graph": {}})
    pair = [fake("a", 1, "same"), fake("b", 2, "same")] + [fake("c", i) for i in range(20)]
    recs = build_records(pair, 0.3, 42, {})
    same = {r["split"] for r in recs if r["source_id"] == "same"}
    ids = [f"q{i}" for i in range(50)]
    cases = [
        ("target is smallest count", target == 3 and all(len(v) == 3 for v in selected.values())),
        ("zero type left out with warning", any("'d'" in w for w in warn)
         and "d" not in balance(samples, ["a", "b", "c", "d"], 0, 42)[0]),
        ("clean sample labels", clean["error_type"] == "none"
         and clean["expected_verdict"] == "supported"),
        ("same source same split", len(same) == 1),
        ("schema fields in order with nulls", all(list(r) == SCHEMA_FIELDS for r in recs)
         and recs[0]["path_edges"] is None and recs[0]["original_triple"] is None),
        ("same seed same split", split_by_question(ids, 0.3, 42)
         == split_by_question(ids, 0.3, 42)),
    ]
    failed = 0
    for name, ok in cases:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


def main() -> None:
    """Build the balanced, split benchmark for one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--per-type", type=int, default=0)
    parser.add_argument("--clean-count", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--spotcheck-per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    bench_dir = BENCH_ROOT / args.run_name
    rag_dir = LIGHTRAG_ROOT / args.run_name
    injected_path = bench_dir / "injected.jsonl"
    mapped_path = rag_dir / "mapped.jsonl"
    if not injected_path.exists():
        sys.exit(f"ERROR: {injected_path} not found. Run src/rag/inject_errors.py first.")
    if not mapped_path.exists():
        sys.exit(f"ERROR: {mapped_path} not found. Run src/rag/map_sentences.py first.")

    injected = load_jsonl(injected_path)
    mapped = load_jsonl(mapped_path)
    answers_path = rag_dir / "answers.jsonl"
    models = {r["id"]: r for r in load_jsonl(answers_path)} if answers_path.exists() else {}
    created = date.today().isoformat()
    provenance_for = {
        r["id"]: {"run_name": args.run_name,
                  "llm_model": r.get("llm_model") or models.get(r["id"], {}).get("llm_model"),
                  "embed_model": r.get("embed_model")
                  or models.get(r["id"], {}).get("embed_model"),
                  "seed": args.seed, "created": created}
        for r in mapped
    }

    selected, target, warnings = balance(injected, ERROR_TYPES, args.per_type, args.seed)
    clean_rows = [make_clean(r) for r in mapped if r.get("label") == "correct"]
    clean, clean_warn = sample_clean(clean_rows, args.clean_count or target, args.seed)
    warnings += clean_warn
    samples = [s for t in sorted(selected) for s in selected[t]] + clean
    if not samples:
        sys.exit("ERROR: no samples to write (no injected samples and no clean rows).")
    records = build_records(samples, args.dev_fraction, args.seed, provenance_for)

    for split in ("dev", "test"):
        with (bench_dir / f"benchmark_{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_spotcheck(records, args.spotcheck_per_type, args.seed, bench_dir / "spotcheck.csv")

    summary = {
        **summarise(records),
        "target_per_type": target,
        "clean_target": args.clean_count or target,
        "warnings": warnings,
        "settings": {k: v for k, v in vars(args).items() if k != "self_test"},
        "created": created,
    }
    (bench_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Target per type: {target}, clean target: {summary['clean_target']}")
    for split in ("dev", "test"):
        print(f"{split}: errors={summary['per_error_type'][split]} clean={summary['clean'][split]}")
        print(f"     question types={summary['per_question_type'][split]}")
    lc = summary["leakage_check"]
    print(f"Leakage check: {'passed' if lc['passed'] else 'FAILED'} "
          f"({lc['source_ids_dev']} dev / {lc['source_ids_test']} test source ids, "
          f"overlap {len(lc['overlap'])})")
    for w in warnings:
        print(f"WARNING: {w}")
    for name in ("benchmark_dev.jsonl", "benchmark_test.jsonl", "spotcheck.csv",
                 "benchmark_summary.json"):
        print(f"Saved -> {bench_dir / name}")


if __name__ == "__main__":
    main()
