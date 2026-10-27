"""Plant known errors into correct LightRAG answers to build the auto-labelled benchmark (Phase C).

Each error type (docs/taxonomy.md, section 3) is one injector function registered in INJECTORS.
An injector looks at one mapped answer (from map_sentences.py) and returns candidate samples;
the main loop validates them and keeps at most one sample per type per answer.
Implemented: relation_reversal, entity_swap. To add a type, write an injector with the same
signature and add it to INJECTORS and EXPECTED_VERDICT.
"""
import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Callable

from entities import TRAILING_PAREN_RE, build_reference_graph, normalise_entity

OUT_ROOT = Path("outputs/lightrag")
BENCH_ROOT = Path("outputs/benchmark")

# From the 762 gold evidence triples in data/questions/questions.jsonl (25 relations).
# (A, r, B) means the same as (B, r, A): reversing these is not an error.
SYMMETRIC_RELATIONS = {"spouse", "sibling"}
# Asymmetric, person to person: the only relations a reversal can apply to.
REVERSIBLE_RELATIONS = {"father", "mother", "child", "doctoral advisor"}

# docs/taxonomy.md, section 5.
EXPECTED_VERDICT = {
    "relation_reversal": "contradicted",
    "entity_swap": "contradicted",
}

Skips = dict[str, int]


def find_span(sentence: str, node: dict) -> tuple[int, int] | None:
    """Locate the node's mention in the original sentence (case-insensitive), or None."""
    spellings = []
    for name in node["names"]:
        for s in (name, TRAILING_PAREN_RE.sub("", name)):
            if s and s not in spellings:
                spellings.append(s)
    lowered = sentence.lower()
    for s in sorted(spellings, key=len, reverse=True):
        pos = lowered.find(s.lower())
        if pos >= 0:
            return pos, pos + len(s)
    return None


def _node(graph: dict, key: str) -> dict:
    """Node dict for a key."""
    return next(n for n in graph["nodes"] if n["key"] == key)


def _skip(skips: Skips, reason: str) -> None:
    skips[reason] = skips.get(reason, 0) + 1


def _sample(row: dict, error_type: str, sentence: dict, new_text: str,
            original: list[str], new: list[str]) -> dict:
    """Build one sample (seed and created are added by the main loop)."""
    answer = row.get("answer") or ""
    if sentence["text"] in answer:
        injected = answer.replace(sentence["text"], new_text, 1)
    else:  # fall back to rebuilding the answer from its sentences
        injected = " ".join(new_text if s["index"] == sentence["index"] else s["text"]
                            for s in row["sentences"])
    return {
        "sample_id": f"{row['id']}__{error_type}",
        "source_id": row["id"],
        "question_type": row.get("type"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "error_type": error_type,
        "expected_verdict": EXPECTED_VERDICT[error_type],
        "sentence_index": sentence["index"],
        "original_sentence": sentence["text"],
        "new_sentence": new_text,
        "original_triple": original,
        "new_triple": new,
        "original_answer": answer,
        "injected_answer": injected,
        "reference_graph": row["reference_graph"],
    }


def _direct_edges(row: dict) -> list[tuple[dict, dict]]:
    """(sentence, edge) pairs for every direct edge stated by a sentence."""
    edges = {e["index"]: e for e in row["reference_graph"]["edges"]}
    return [(s, edges[i]) for s in row["sentences"] for i in s["edges"]]


def inject_relation_reversal(row: dict, rng: random.Random,
                             pools: dict[str, list[str]]) -> tuple[list[dict], Skips]:
    """Swap subject and object mentions of a reversible direct edge."""
    graph, out, skips = row["reference_graph"], [], {}
    pairs = _direct_edges(row)
    if not pairs:
        _skip(skips, "no_direct_edge")
    for sent, e in pairs:
        if e["relation"] not in REVERSIBLE_RELATIONS:
            _skip(skips, "relation_not_reversible")
            continue
        s_span = find_span(sent["text"], _node(graph, e["subject_key"]))
        o_span = find_span(sent["text"], _node(graph, e["object_key"]))
        if s_span is None or o_span is None:
            _skip(skips, "span_not_found")
            continue
        (a0, a1), (b0, b1) = sorted([s_span, o_span])
        if a1 > b0:
            _skip(skips, "span_overlap")
            continue
        t = sent["text"]
        new_text = t[:a0] + t[b0:b1] + t[a1:b0] + t[a0:a1] + t[b1:]
        out.append(_sample(row, "relation_reversal", sent, new_text,
                           [e["subject"], e["relation"], e["object"]],
                           [e["object"], e["relation"], e["subject"]]))
    return out, skips


def inject_entity_swap(row: dict, rng: random.Random,
                       pools: dict[str, list[str]]) -> tuple[list[dict], Skips]:
    """Replace the object of a direct edge with another object of the same relation."""
    graph, out, skips = row["reference_graph"], [], {}
    graph_keys = {n["key"] for n in graph["nodes"]}
    pairs = _direct_edges(row)
    if not pairs:
        _skip(skips, "no_direct_edge")
    for sent, e in pairs:
        span = find_span(sent["text"], _node(graph, e["object_key"]))
        if span is None:
            _skip(skips, "span_not_found")
            continue
        options = [o for o in pools.get(e["relation"], [])
                   if normalise_entity(o) not in graph_keys]
        if not options:
            _skip(skips, "empty_pool")
            continue
        new_obj = rng.choice(options)
        t = sent["text"]
        new_text = t[:span[0]] + new_obj + t[span[1]:]
        out.append(_sample(row, "entity_swap", sent, new_text,
                           [e["subject"], e["relation"], e["object"]],
                           [e["subject"], e["relation"], new_obj]))
    return out, skips


Injector = Callable[[dict, random.Random, dict[str, list[str]]], tuple[list[dict], Skips]]
INJECTORS: dict[str, Injector] = {
    "relation_reversal": inject_relation_reversal,
    "entity_swap": inject_entity_swap,
}


def validate(sample: dict, graph: dict) -> bool:
    """New triple must not be a gold edge (normalised keys) and the sentence must change."""
    s, r, o = sample["new_triple"]
    existing = {(e["subject_key"], e["relation"], e["object_key"]) for e in graph["edges"]}
    return ((normalise_entity(s), r, normalise_entity(o)) not in existing
            and sample["new_sentence"] != sample["original_sentence"])


def build_pools(questions: list[dict]) -> dict[str, list[str]]:
    """For each relation, all distinct objects (by normalised key) across all questions."""
    pools: dict[str, dict[str, str]] = {}
    for q in questions:
        for _, rel, obj in q["evidences"]:
            pools.setdefault(rel, {}).setdefault(normalise_entity(obj), obj)
    return {rel: sorted(objs.values()) for rel, objs in pools.items()}


def inject_row(row: dict, error_type: str, seed: int, pools: dict[str, list[str]],
               skips: Skips) -> tuple[dict | None, int]:
    """Run one injector on one row; return (chosen sample or None, valid candidate count)."""
    rng = random.Random(f"{seed}:{row['id']}:{error_type}")
    candidates, injector_skips = INJECTORS[error_type](row, rng, pools)
    for reason, n in injector_skips.items():
        skips[reason] = skips.get(reason, 0) + n
    valid = []
    for c in candidates:
        if validate(c, row["reference_graph"]):
            valid.append(c)
        else:
            _skip(skips, "invalid")
    return (rng.choice(valid) if valid else None), len(valid)


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_self_test() -> None:
    """Run built-in cases, print PASS/FAIL, exit 1 if any fail."""
    from map_sentences import map_answer

    def make_row(evidences: list[list[str]], answer: str) -> dict:
        graph = build_reference_graph(evidences)
        return {"id": "t", "type": "test", "question": "", "gold_answer": "", "answer": answer,
                "label": "correct", "reference_graph": graph, **map_answer(answer, graph, 2)}

    pools = {"date of birth": ["June 2, 1960", "May 1, 1950"]}
    rev = make_row([["Anne Smith", "father", "John Smith"]], "Anne Smith's father is John Smith.")
    rev_s, _ = inject_row(rev, "relation_reversal", 42, pools, {})
    director, d_skips = inject_relation_reversal(
        make_row([["Some Film", "director", "Jane Doe"]], "Some Film was directed by Jane Doe."),
        random.Random(1), pools)
    spouse, s_skips = inject_relation_reversal(
        make_row([["Anne Smith", "spouse", "Tom Brown"]], "Anne Smith married Tom Brown."),
        random.Random(1), pools)
    dob = make_row([["Anne Smith", "date of birth", "May 1, 1950"]],
                   "Anne Smith was born on May 1, 1950.")
    swap, _ = inject_row(dob, "entity_swap", 42, pools, {})
    dob_keys = {n["key"] for n in dob["reference_graph"]["nodes"]}
    existing = dict(swap or {}, new_triple=["Anne Smith", "date of birth", "May 1, 1950"])
    other_order = make_row([["Anne Smith", "date of birth", "May 1, 1950"]],
                           "Anne Smith was born on 1 May 1950.")
    nf_samples, nf_skips = inject_entity_swap(other_order, random.Random(1), pools)
    cases = [
        ("reversal sentence", rev_s is not None
         and rev_s["new_sentence"] == "John Smith's father is Anne Smith."),
        ("reversal triple", rev_s is not None
         and rev_s["new_triple"] == ["John Smith", "father", "Anne Smith"]),
        ("no reversal for director", director == []
         and d_skips.get("relation_not_reversible") == 1),
        ("no reversal for spouse", spouse == [] and s_skips.get("relation_not_reversible") == 1),
        ("entity swap date", swap is not None
         and swap["new_sentence"] == "Anne Smith was born on June 2, 1960."
         and normalise_entity(swap["new_triple"][2]) not in dob_keys),
        ("validate rejects existing edge", swap is not None
         and not validate(existing, dob["reference_graph"])),
        ("span_not_found", other_order["sentences"][0]["edges"] == [0] and nf_samples == []
         and nf_skips.get("span_not_found") == 1),
    ]
    failed = 0
    for name, ok in cases:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


def main() -> None:
    """Inject errors into every correct mapped answer of a run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument("--questions", type=Path, default=Path("data/questions/questions.jsonl"))
    parser.add_argument("--types", default="relation_reversal,entity_swap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    unknown = [t for t in types if t not in INJECTORS]
    if unknown:
        sys.exit(f"ERROR: unknown error type(s) {unknown}. Available: {sorted(INJECTORS)}")

    mapped_path = OUT_ROOT / args.run_name / "mapped.jsonl"
    if not mapped_path.exists():
        sys.exit(f"ERROR: {mapped_path} not found. Run src/rag/map_sentences.py first.")
    if not args.questions.exists():
        sys.exit(f"ERROR: {args.questions} not found. Run src/data/prepare_questions.py first.")
    rows = [r for r in load_jsonl(mapped_path) if r.get("label") == "correct"]
    pools = build_pools(load_jsonl(args.questions))

    created = date.today().isoformat()
    samples: list[dict] = []
    per_type = {t: 0 for t in types}
    candidates = {t: 0 for t in types}
    skips: dict[str, Skips] = {t: {} for t in types}
    for row in rows:
        for t in types:
            sample, n_valid = inject_row(row, t, args.seed, pools, skips[t])
            candidates[t] += n_valid
            if sample is None:
                _skip(skips[t], "no_valid_candidate")
                continue
            samples.append({**sample, "seed": args.seed, "created": created})
            per_type[t] += 1

    out_dir = BENCH_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "injected.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = {
        "correct_rows_read": len(rows),
        "samples_per_type": per_type,
        "valid_candidates_per_type": candidates,
        "skips_per_type": {t: dict(sorted(s.items())) for t, s in skips.items()},
        "symmetric_relations": sorted(SYMMETRIC_RELATIONS),
        "reversible_relations": sorted(REVERSIBLE_RELATIONS),
        "seed": args.seed,
        "created": created,
    }
    (out_dir / "inject_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Correct rows read: {len(rows)} from {mapped_path}")
    for t in types:
        print(f"  {t:<20} samples={per_type[t]}  valid candidates={candidates[t]}  "
              f"skips={skips[t]}")
    print(f"Symmetric: {sorted(SYMMETRIC_RELATIONS)}")
    print(f"Reversible: {sorted(REVERSIBLE_RELATIONS)}")
    print(f"Saved -> {out_dir / 'injected.jsonl'}")
    print(f"Saved -> {out_dir / 'inject_summary.json'}")


if __name__ == "__main__":
    main()
