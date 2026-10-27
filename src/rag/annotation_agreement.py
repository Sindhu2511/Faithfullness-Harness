"""Compare two annotators' sheets, measure agreement and merge into final labels (tasks D4-D5).

Reads the filled sheets written by make_annotation_set.py (labels follow
docs/annotation_guideline.md), validates every row, reports raw agreement and Cohen's kappa for
verdict, error_type and a binary hallucinated label, writes the disagreements for adjudication,
and merges agreed plus adjudicated rows into final sentence- and answer-level labels.
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ANNOT_ROOT = Path("outputs/annotation")
VERDICTS = ["supported", "contradicted", "unsupported", "no_claim"]
ERROR_TYPES = ["none", "relation_reversal", "wrong_relation", "entity_swap", "broken_chain",
               "fabricated_fact"]
CONTRADICTION_TYPES = {"relation_reversal", "wrong_relation", "entity_swap", "broken_chain"}
HALLUCINATED = {"contradicted", "unsupported"}
LABEL_COLUMNS = ["verdict", "error_type", "evidence_edges"]


def parse_edges(text: str | None) -> list[int] | None:
    """Comma-separated edge indexes; [] for blank, None if not parseable."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        return sorted({int(p.strip()) for p in text.split(",")})
    except ValueError:
        return None


def validate_labels(verdict: str | None, error_type: str | None,
                    evidence: str | None) -> tuple[dict | None, str | None]:
    """Return (labels, None) if valid, else (None, reason). Values compared stripped, lowercase."""
    v = (verdict or "").strip().lower()
    e = (error_type or "").strip().lower()
    if not v:
        return None, "verdict blank"
    if v not in VERDICTS:
        return None, f"verdict '{v}' not allowed"
    if not e:
        return None, "error_type blank"
    if e not in ERROR_TYPES:
        return None, f"error_type '{e}' not allowed"
    if v in ("supported", "no_claim") and e != "none":
        return None, f"verdict {v} needs error_type none"
    if v == "contradicted" and e not in CONTRADICTION_TYPES:
        return None, f"verdict contradicted cannot have error_type {e}"
    if v == "unsupported" and e != "fabricated_fact":
        return None, "verdict unsupported needs error_type fabricated_fact"
    edges = parse_edges(evidence)
    if edges is None:
        return None, f"evidence_edges '{(evidence or '').strip()}' not parseable"
    return {"verdict": v, "error_type": e, "evidence_edges": edges}, None


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float | None, str | None]:
    """Cohen's kappa = (po - pe) / (1 - pe); returns (kappa, note)."""
    n = len(a)
    if n == 0:
        return None, "no rows to compare"
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if pe == 1:
        return (1.0 if po == 1 else 0.0), "expected agreement is 1 (a single label used); kappa undefined"
    return (po - pe) / (1 - pe), None


def band(kappa: float | None) -> str | None:
    """Landis and Koch band."""
    if kappa is None:
        return None
    if kappa < 0:
        return "poor"
    for upper, name in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                        (0.80, "substantial")):
        if kappa <= upper:
            return name
    return "almost perfect"


def agreement(a: list[str], b: list[str]) -> dict:
    """Raw agreement, kappa, band and note for one label set."""
    k, note = cohen_kappa(a, b)
    raw = sum(x == y for x, y in zip(a, b)) / len(a) if a else None
    return {"raw_agreement": round(raw, 3) if raw is not None else None,
            "kappa": round(k, 3) if k is not None else None, "band": band(k), "note": note}


def confusion(a: list[str], b: list[str]) -> dict[str, dict[str, int]]:
    """Verdict confusion matrix: first annotator as rows, second as columns."""
    m = {x: {y: 0 for y in VERDICTS} for x in VERDICTS}
    for x, y in zip(a, b):
        m[x][y] += 1
    return m


def final_label(a: dict, b: dict, adjudicated: dict | None) -> tuple[dict | None, str]:
    """Final labels for one row from two valid label dicts and an optional adjudication row."""
    if a["verdict"] == b["verdict"] and a["error_type"] == b["error_type"]:
        return {"verdict": a["verdict"], "error_type": a["error_type"],
                "evidence_edges": sorted(set(a["evidence_edges"]) | set(b["evidence_edges"]))}, \
            "agreed"
    if adjudicated:
        labels, _ = validate_labels(adjudicated.get("final_verdict"),
                                    adjudicated.get("final_error_type"),
                                    adjudicated.get("final_evidence_edges"))
        if labels:
            return labels, "adjudicated"
    return None, "unresolved"


def answer_level(answer_id: str, labels: list[dict], meta: dict) -> dict:
    """Answer-level summary from its resolved sentence labels."""
    counts = {v: sum(1 for l in labels if l["verdict"] == v) for v in VERDICTS}
    judged = counts["supported"] + counts["contradicted"] + counts["unsupported"]
    return {
        "answer_id": answer_id,
        "question_type": meta.get("question_type"),
        "split": meta.get("split"),
        "n_sentences": len(labels),
        "verdict_counts": counts,
        "hallucinated": counts["contradicted"] + counts["unsupported"] > 0,
        "faithfulness_score": round(counts["supported"] / judged, 3) if judged else None,
        "error_types": sorted({l["error_type"] for l in labels} - {"none"}),
    }


def read_csv(path: Path) -> list[dict]:
    """Read a CSV that Excel may have saved with a BOM."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_self_test() -> None:
    """Run built-in cases, print PASS/FAIL, exit 1 if any fail."""
    a_lab, _ = validate_labels("contradicted", "entity_swap", "0")
    b_lab, _ = validate_labels("Contradicted", "Entity_Swap", "1, 0")
    agreed, agreed_src = final_label(a_lab, b_lab, None)
    other, _ = validate_labels("supported", "none", "0")
    _, unresolved_src = final_label(a_lab, other, None)
    adj, adj_src = final_label(a_lab, other, {"final_verdict": "supported",
                                              "final_error_type": "none",
                                              "final_evidence_edges": "0"})
    ans = answer_level("q1", [{"verdict": "supported", "error_type": "none"},
                              {"verdict": "contradicted", "error_type": "entity_swap"},
                              {"verdict": "no_claim", "error_type": "none"}], {})
    cases = [
        ("kappa identical is 1.0", cohen_kappa(["a", "b", "c"], ["a", "b", "c"])[0] == 1.0),
        ("kappa example is 0.5",
         abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"])[0] - 0.5) < 1e-9),
        ("contradicted + none invalid", validate_labels("contradicted", "none", "")[0] is None),
        ("unsupported + fabricated_fact valid",
         validate_labels("unsupported", "fabricated_fact", "")[0] is not None),
        ("evidence parsing", parse_edges(" 0, 2 ") == [0, 2] and parse_edges("x") is None),
        ("agreed row", agreed_src == "agreed" and agreed["evidence_edges"] == [0, 1]),
        ("unresolved then adjudicated", unresolved_src == "unresolved"
         and adj_src == "adjudicated" and adj["verdict"] == "supported"),
        ("answer-level score", ans["faithfulness_score"] == 0.5 and ans["hallucinated"] is True),
    ]
    failed = 0
    for name, ok in cases:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


def main() -> None:
    """Validate, measure agreement, write disagreements and final labels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--annotators", default="A,B")
    parser.add_argument("--adjudicated", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    names = [n.strip() for n in args.annotators.split(",") if n.strip()]
    if len(names) != 2:
        sys.exit(f"ERROR: --annotators needs exactly two names, got {names}.")
    na, nb = names
    out_dir = ANNOT_ROOT / args.run_name
    items_path = out_dir / "annotation_items.jsonl"
    if not items_path.exists():
        sys.exit(f"ERROR: {items_path} not found. Run src/rag/make_annotation_set.py first.")
    sheets = {}
    for n in names:
        path = out_dir / f"sheet_{n}.csv"
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Annotator {n} must fill and save this sheet "
                     "(see docs/annotation_guideline.md).")
        rows = read_csv(path)
        missing = {"row_id", *LABEL_COLUMNS} - set(rows[0] if rows else {})
        if missing:
            sys.exit(f"ERROR: {path} is missing columns {sorted(missing)}; do not rename columns.")
        sheets[n] = {r["row_id"]: r for r in rows}
    adjudicated = {}
    if args.adjudicated:
        if not args.adjudicated.exists():
            sys.exit(f"ERROR: {args.adjudicated} not found. Fill final_verdict, final_error_type "
                     "and final_evidence_edges in disagreements.csv first.")
        adjudicated = {r["row_id"]: r for r in read_csv(args.adjudicated)}
    items = {r["row_id"]: r for r in load_jsonl(items_path)}

    # 1. validation
    valid: dict[str, dict[str, dict]] = {n: {} for n in names}
    invalid = []
    for n in names:
        for rid, r in sheets[n].items():
            labels, reason = validate_labels(r.get("verdict"), r.get("error_type"),
                                             r.get("evidence_edges"))
            if labels:
                valid[n][rid] = labels
            else:
                invalid.append({"sheet": n, "row_id": rid, "reason": reason})
    only_one = [{"row_id": rid, "only_in": na if rid in sheets[na] else nb}
                for rid in sorted(set(sheets[na]) ^ set(sheets[nb]))]
    both = [rid for rid in items if rid in valid[na] and rid in valid[nb]]

    # 2. agreement
    va = [valid[na][r]["verdict"] for r in both]
    vb = [valid[nb][r]["verdict"] for r in both]
    ea = [valid[na][r]["error_type"] for r in both]
    eb = [valid[nb][r]["error_type"] for r in both]
    ha = [str(v in HALLUCINATED) for v in va]
    hb = [str(v in HALLUCINATED) for v in vb]
    agree = {"verdict": agreement(va, vb), "error_type": agreement(ea, eb),
             "hallucinated_binary": agreement(ha, hb)}

    # 3-4. disagreements and final labels
    disagreements, finals = [], []
    n_adjudicated = 0
    for rid in both:
        a, b = valid[na][rid], valid[nb][rid]
        labels, source = final_label(a, b, adjudicated.get(rid))
        if source != "agreed":
            item = items[rid]
            # Keep adjudication already filled in, so rerunning on disagreements.csv is safe.
            prev = adjudicated.get(rid, {})
            disagreements.append({
                "row_id": rid, "question": item.get("question"), "sentence": item.get("sentence"),
                "reference_triples": item.get("reference_triples"),
                f"verdict_{na}": a["verdict"], f"error_type_{na}": a["error_type"],
                f"evidence_{na}": ",".join(map(str, a["evidence_edges"])),
                f"verdict_{nb}": b["verdict"], f"error_type_{nb}": b["error_type"],
                f"evidence_{nb}": ",".join(map(str, b["evidence_edges"])),
                **{c: prev.get(c) or "" for c in ("final_verdict", "final_error_type",
                                                    "final_evidence_edges", "notes")},
            })
        n_adjudicated += source == "adjudicated"
        if labels:
            item = items[rid]
            finals.append({
                "row_id": rid, "answer_id": item.get("answer_id"),
                "question_type": item.get("question_type"), "question": item.get("question"),
                "gold_answer": item.get("gold_answer"), "sentence_index": item.get("sentence_index"),
                "sentence": item.get("sentence"), **labels, "source": source,
                "split": item.get("split"), "reference_graph": item.get("reference_graph"),
            })

    dis_cols = ["row_id", "question", "sentence", "reference_triples",
                f"verdict_{na}", f"error_type_{na}", f"evidence_{na}",
                f"verdict_{nb}", f"error_type_{nb}", f"evidence_{nb}",
                "final_verdict", "final_error_type", "final_evidence_edges", "notes"]
    with (out_dir / "disagreements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dis_cols)
        w.writeheader()
        w.writerows(disagreements)
    with (out_dir / "final_labels.jsonl").open("w", encoding="utf-8") as f:
        for r in finals:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # answer level: only answers whose sentences are all resolved
    by_answer: dict[str, list[str]] = {}
    for rid, item in items.items():
        by_answer.setdefault(item["answer_id"], []).append(rid)
    final_by_row = {r["row_id"]: r for r in finals}
    answers = [answer_level(aid, [final_by_row[r] for r in rids], items[rids[0]])
               for aid, rids in by_answer.items() if all(r in final_by_row for r in rids)]
    with (out_dir / "final_answers.jsonl").open("w", encoding="utf-8") as f:
        for r in answers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. summary
    summary = {
        "rows_per_sheet": {n: len(sheets[n]) for n in names},
        "valid_rows": {n: len(valid[n]) for n in names},
        "rows_valid_in_both": len(both),
        "invalid_rows": invalid,
        "rows_in_one_sheet_only": only_one,
        "agreement": agree,
        "verdict_confusion_matrix": {"rows": na, "columns": nb, "matrix": confusion(va, vb)},
        "disagreements": sum(1 for rid in both if final_label(valid[na][rid], valid[nb][rid],
                                                              None)[1] != "agreed"),
        "adjudicated": n_adjudicated,
        "unresolved": len(both) - len(finals),
        "final_rows": len(finals),
        "final_answers": len(answers),
        "final_per_verdict": {v: sum(r["verdict"] == v for r in finals) for v in VERDICTS},
        "final_per_error_type": {e: sum(r["error_type"] == e for r in finals)
                                 for e in ERROR_TYPES},
        "adjudicated_file": str(args.adjudicated) if args.adjudicated else None,
        "created": date.today().isoformat(),
    }
    (out_dir / "agreement_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Rows: {na}={len(sheets[na])}, {nb}={len(sheets[nb])}, valid in both={len(both)}, "
          f"invalid={len(invalid)}, in one sheet only={len(only_one)}")
    for key, a in agree.items():
        print(f"  {key:<20} raw={a['raw_agreement']}  kappa={a['kappa']}  ({a['band']})"
              + (f"  note: {a['note']}" if a["note"] else ""))
    print(f"Disagreements: {summary['disagreements']}, adjudicated: {n_adjudicated}, "
          f"unresolved: {summary['unresolved']}")
    print(f"Final: {len(finals)} sentence labels, {len(answers)} fully resolved answers")
    for name in ("disagreements.csv", "final_labels.jsonl", "final_answers.jsonl",
                 "agreement_summary.json"):
        print(f"Saved -> {out_dir / name}")


if __name__ == "__main__":
    main()
