"""Select a fixed, stratified set of 2WikiMultiHopQA questions for the benchmark (tasks B1-B2).

Why 2WikiMultiHopQA: every question is multi-hop, and each one ships with gold evidence
triples (subject, relation, object) that form its reasoning path, which matches the shape of
our graph. Each question also ships with its context paragraphs as plain text, which LightRAG
can ingest to build its graph. Source: Hugging Face dataset framolfese/2WikiMultihopQA.
The output of this script is the question set the LightRAG step (B3) runs on.
"""
import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

from datasets import Dataset, load_dataset

HF_REPO_ID = "framolfese/2WikiMultihopQA"


def load_split(split: str) -> Dataset:
    """Load one split from Hugging Face (cached by the datasets library)."""
    try:
        ds = load_dataset(HF_REPO_ID, split=split)
    except Exception as e:  # network error, unknown split, etc.
        sys.exit(f"ERROR: could not load split '{split}' from {HF_REPO_ID}: {e}")
    if split == "test":
        answers = ds["answer"]
        empty = sum(1 for a in answers if not (a or "").strip())
        if empty > len(answers) / 2:
            sys.exit(f"ERROR: {empty}/{len(answers)} answers in the test split are empty "
                     "(test answers are hidden). Use --split validation.")
    return ds


def filter_rows(ds: Dataset, min_evidences: int) -> list[int]:
    """Return indices of rows with >= min_evidences evidence triples and a non-empty answer."""
    return [
        i for i, (ev, ans) in enumerate(zip(ds["evidences"], ds["answer"]))
        if len(ev or []) >= min_evidences and (ans or "").strip()
    ]


def stratified_sample(types: list[str], indices: list[int], n: int,
                      seed: int) -> tuple[list[int], list[str]]:
    """Take n // number_of_types indices per type; return the sample and notes on short types."""
    groups: dict[str, list[int]] = {}
    for i in indices:
        groups.setdefault(types[i], []).append(i)
    per_type = n // len(groups)
    rng = random.Random(seed)
    sample, notes = [], []
    for t in sorted(groups):
        group = groups[t]
        if len(group) < per_type:
            notes.append(f"type '{t}' has only {len(group)} rows (wanted {per_type}); took all")
            sample.extend(group)
        else:
            sample.extend(rng.sample(group, per_type))
    if per_type * len(groups) != n:
        notes.append(f"{n} is not divisible by {len(groups)} types; took {per_type} per type")
    return sample, notes


def to_record(row: dict) -> dict:
    """Convert one dataset row to the benchmark question format."""
    # context and supporting_facts are column-wise dicts of parallel lists; zip them row-wise.
    ctx, sf = row["context"], row["supporting_facts"]
    return {
        "id": row["id"],
        "type": row["type"],
        "question": row["question"],
        "gold_answer": row["answer"],
        "evidences": [list(e) for e in row["evidences"]],
        "context": [{"title": t, "sentences": list(s)}
                    for t, s in zip(ctx["title"], ctx["sentences"])],
        "supporting_facts": [{"title": t, "sent_id": int(i)}
                             for t, i in zip(sf["title"], sf["sent_id"])],
    }


def write_outputs(records: list[dict], summary: dict, out_dir: Path) -> None:
    """Write questions.jsonl and questions_summary.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "questions_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Load, filter, sample and save the question set plus a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-evidences", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("data/questions"))
    args = parser.parse_args()

    ds = load_split(args.split)
    kept = filter_rows(ds, args.min_evidences)
    if not kept:
        sys.exit(f"ERROR: no rows left after keeping those with >= {args.min_evidences} "
                 f"evidence triples and a non-empty answer ({len(ds)} loaded).")

    sample, notes = stratified_sample(ds["type"], kept, args.n, args.seed)
    records = [to_record(row) for row in ds.select(sample)]

    per_type: dict[str, int] = {}
    for r in records:
        per_type[r["type"]] = per_type.get(r["type"], 0) + 1
    summary = {
        "total": len(records),
        "per_type": dict(sorted(per_type.items())),
        "rows_before_filter": len(ds),
        "rows_after_filter": len(kept),
        "seed": args.seed,
        "split": args.split,
        "min_evidences": args.min_evidences,
        "source_repo_id": HF_REPO_ID,
        "notes": notes,
        "created": date.today().isoformat(),
    }
    write_outputs(records, summary, args.out_dir)

    print(f"\nSource: {HF_REPO_ID}, split={args.split}, seed={args.seed}")
    print(f"Rows: {len(ds)} loaded, {len(kept)} after filter "
          f"(>= {args.min_evidences} evidences, non-empty answer)")
    print(f"Sampled {len(records)}:")
    for t, c in summary["per_type"].items():
        print(f"  {t:<20}{c:>5}")
    for note in notes:
        print(f"NOTE: {note}")
    print(f"Saved -> {args.out_dir / 'questions.jsonl'}")
    print(f"Saved -> {args.out_dir / 'questions_summary.json'}")


if __name__ == "__main__":
    main()
