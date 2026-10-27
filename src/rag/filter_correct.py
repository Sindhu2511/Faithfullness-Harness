"""Judge LightRAG answers against gold answers (task B4).

Correct answers become the clean base for error injection (Phase C). Incorrect answers are
kept for the human-annotated set (Phase D). Comparison answers that name both options are
sent to human review, since string matching cannot tell which option was chosen.
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

OUT_ROOT = Path("outputs/lightrag")
ARTICLES = {"a", "an", "the"}
COMPARISON_TYPES = {"comparison", "bridge_comparison"}
# ", <option A> or <option B>?" at the end of the question. Leftmost comma, first " or ".
OPTIONS_RE = re.compile(r",\s*(.+?)\s+or\s+(.+?)\s*\?\s*$", re.IGNORECASE)
REVIEW_COLUMNS = ["id", "type", "question", "gold_answer", "answer", "human_label"]


def normalise(text: str) -> str:
    """NFKD + strip accents, lowercase, drop punctuation and articles, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(w for w in text.split() if w not in ARTICLES)


def tokens(text: str) -> list[str]:
    """Normalised word tokens."""
    return normalise(text).split()


def contains_words(needle: str, haystack: str) -> bool:
    """Whether normalised `needle` appears in normalised `haystack` as whole words."""
    return bool(needle) and f" {needle} " in f" {haystack} "


def extract_options(question: str) -> tuple[str, str] | None:
    """The two options of a comparison question, or None if the pattern does not match."""
    m = OPTIONS_RE.search(question or "")
    return (m.group(1), m.group(2)) if m else None


def judge_row(row: dict, threshold: float) -> tuple[str, str, float | None]:
    """Return (label, rule, recall) for one answer row."""
    if row.get("error") is not None:
        return "error", "error", None

    gold, answer = normalise(row.get("gold_answer", "")), normalise(row.get("answer") or "")
    gold_tok, ans_tok = gold.split(), answer.split()
    recall = (sum(1 for t in set(gold_tok) if t in set(ans_tok)) / len(set(gold_tok))
              if gold_tok else None)

    if gold in ("yes", "no"):
        opposite = "no" if gold == "yes" else "yes"
        first_ok = bool(ans_tok) and ans_tok[0] == gold
        mention_ok = gold in ans_tok and opposite not in ans_tok
        return ("correct" if first_ok or mention_ok else "incorrect"), "yes_no", recall

    if contains_words(gold, answer):
        options = extract_options(row.get("question", "")) \
            if row.get("type") in COMPARISON_TYPES else None
        if options and all(contains_words(normalise(o), answer) for o in options):
            return "needs_review", "comparison_ambiguity", recall
        return "correct", "exact_containment", recall

    if recall is not None and recall >= threshold:
        return "correct", "token_recall", recall
    return "incorrect", "token_recall", recall


def load_rows(path: Path) -> list[dict]:
    """Load answers.jsonl."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run src/rag/run_lightrag.py first.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_review(path: Path) -> dict[str, str]:
    """Map id -> human_label ("correct"/"incorrect") from a filled review CSV."""
    if not path.exists():
        sys.exit(f"ERROR: review file {path} not found.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        labels = {r["id"]: (r.get("human_label") or "").strip().lower() for r in csv.DictReader(f)}
    return {i: lab for i, lab in labels.items() if lab in ("correct", "incorrect")}


def apply_review(rows: list[dict], human: dict[str, str]) -> int:
    """Override automatic labels with human labels (rule "human"); return how many changed."""
    n = 0
    for r in rows:
        if r["id"] in human:
            r["label"], r["rule"] = human[r["id"]], "human"
            n += 1
    return n


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_outputs(rows: list[dict], review_rows: list[dict], human: dict[str, str],
                  summary: dict, out_dir: Path) -> None:
    """Write judged/clean/incorrect JSONL, review.csv and filter_summary.json."""
    _write_jsonl(out_dir / "judged.jsonl", rows)
    _write_jsonl(out_dir / "clean.jsonl", [r for r in rows if r["label"] == "correct"])
    _write_jsonl(out_dir / "incorrect.jsonl", [r for r in rows if r["label"] == "incorrect"])
    with (out_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in review_rows:
            # Keep human labels already given, so a filled review.csv is not wiped on re-run.
            w.writerow({**{k: r.get(k, "") for k in REVIEW_COLUMNS[:-1]},
                        "human_label": human.get(r["id"], "")})
    (out_dir / "filter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_self_test() -> None:
    """Run built-in cases, print PASS/FAIL, exit 1 if any fail."""
    q_bc = ("Which film has the director who was born first, "
            "Slunce, Seno, Jahody or To Sleep With Anger?")
    cases = [
        ({"gold_answer": "To Sleep With Anger", "answer": "The film is To Sleep with Anger."},
         "correct"),
        ({"gold_answer": "no", "answer": "No, they are from different countries."}, "correct"),
        ({"gold_answer": "yes", "answer": "No, they were not."}, "incorrect"),
        ({"gold_answer": "Zdeněk Troška", "answer": "It was directed by Zdenek Troska."},
         "correct"),
        ({"gold_answer": "April 13, 1944", "answer": "He was born on 13 April 1944."}, "correct"),
        ({"type": "bridge_comparison", "question": q_bc, "gold_answer": "To Sleep With Anger",
          "answer": "Slunce, Seno, Jahody was directed earlier than To Sleep With Anger."},
         "needs_review"),
        ({"gold_answer": "1944", "answer": "", "error": "timeout"}, "error"),
        ({"type": "comparison", "question": "Who was born first, Alice Smith or Bob Jones?",
          "gold_answer": "Alice Smith", "answer": "Alice Smith was born first."}, "correct"),
        ({"gold_answer": "Charles Burnett", "answer": "It was Martin Scorsese."}, "incorrect"),
        ({"gold_answer": "yes", "answer": "Both films are American, so yes."}, "correct"),
    ]
    failed = 0
    for k, (row, expected) in enumerate(cases, start=1):
        row = {"type": "compositional", "question": "", "error": None, **row}
        label, rule, recall = judge_row(row, 0.8)
        ok = label == expected
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {k}: expected {expected}, got {label} "
              f"(rule={rule}, recall={recall})")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


def main() -> None:
    """Judge every answer of a run and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--recall-threshold", type=float, default=0.8)
    parser.add_argument("--review-file", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    out_dir = OUT_ROOT / args.run_name
    rows = load_rows(out_dir / "answers.jsonl")
    for r in rows:
        r["label"], r["rule"], recall = judge_row(r, args.recall_threshold)
        r["recall"] = round(recall, 3) if recall is not None else None
    review_rows = [r for r in rows if r["label"] == "needs_review"]

    human = read_review(args.review_file) if args.review_file else {}
    n_human = apply_review(rows, human)

    per_label: dict[str, int] = {}
    per_rule: dict[str, int] = {}
    per_type: dict[str, dict[str, int]] = {}
    for r in rows:
        per_label[r["label"]] = per_label.get(r["label"], 0) + 1
        per_rule[r["rule"]] = per_rule.get(r["rule"], 0) + 1
        t = per_type.setdefault(r.get("type", "unknown"), {})
        t[r["label"]] = t.get(r["label"], 0) + 1
    summary = {
        "total": len(rows),
        "per_label": dict(sorted(per_label.items())),
        "per_label_per_type": {t: dict(sorted(c.items())) for t, c in sorted(per_type.items())},
        "per_rule": dict(sorted(per_rule.items())),
        "recall_threshold": args.recall_threshold,
        "review_file": str(args.review_file) if args.review_file else None,
        "human_overrides": n_human,
        "created": date.today().isoformat(),
    }
    write_outputs(rows, review_rows, human, summary, out_dir)

    print(f"Judged {len(rows)} answers from {out_dir / 'answers.jsonl'}")
    print("Per label:")
    for k, v in summary["per_label"].items():
        print(f"  {k:<14}{v:>5}")
    print("Per type:")
    for t, c in summary["per_label_per_type"].items():
        print(f"  {t:<20}" + ", ".join(f"{k}={v}" for k, v in c.items()))
    print("Per rule:")
    for k, v in summary["per_rule"].items():
        print(f"  {k:<22}{v:>5}")
    if args.review_file:
        print(f"Human overrides applied: {n_human} (from {args.review_file})")
    print(f"Needs review: {len(review_rows)} -> {out_dir / 'review.csv'}")
    for name in ("judged.jsonl", "clean.jsonl", "incorrect.jsonl", "filter_summary.json"):
        print(f"Saved -> {out_dir / name}")


if __name__ == "__main__":
    main()
