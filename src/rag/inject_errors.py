"""Plant known errors into correct LightRAG answers to build the auto-labelled benchmark (Phase C).

Each error type (docs/taxonomy.md, section 3) is one injector function registered in INJECTORS.
An injector looks at one mapped answer (from map_sentences.py) and returns candidate samples;
the main loop keeps only candidates whose new triple classify_claim() puts in the intended type
(taxonomy sections 3-4), then picks at most one sample per type per answer.
Implemented: relation_reversal, entity_swap, wrong_relation, fabricated_fact, broken_chain.
To add a type, write an injector with the same signature and add it to INJECTORS and
EXPECTED_VERDICT.
"""
import argparse
import json
import random
import re
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

# Only relations in the same group may replace each other (wrong_relation), so both entities
# still make sense. Relations in no group (country, country of origin, country of citizenship,
# occupation, employer, publisher, founded by) are never used; "country" and
# "country of origin" are near-synonyms, so swapping them would not be an error.
FILM_PERSON = {"director", "producer", "composer", "performer", "editor"}
PERSON_PERSON = {"father", "mother", "spouse", "child", "sibling", "doctoral advisor"}
PERSON_DATE = {"date of birth", "date of death"}
WORK_DATE = {"publication date", "inception"}
PERSON_PLACE = {"place of birth", "place of death", "place of burial"}
RELATION_GROUPS = {
    "FILM_PERSON": FILM_PERSON, "PERSON_PERSON": PERSON_PERSON, "PERSON_DATE": PERSON_DATE,
    "WORK_DATE": WORK_DATE, "PERSON_PLACE": PERSON_PLACE,
}
# Relations a node of each kind can be the subject of (fabricated_fact).
KIND_RELATIONS = {
    "work": FILM_PERSON | WORK_DATE,
    "person": PERSON_PERSON | PERSON_DATE | PERSON_PLACE,
}

# One sentence per grouped relation; {s} is the subject, {o} the object.
TEMPLATES = {
    "director": "{s} was directed by {o}.",
    "producer": "{s} was produced by {o}.",
    "composer": "The music of {s} was composed by {o}.",
    "performer": "{s} was performed by {o}.",
    "editor": "{s} was edited by {o}.",
    "father": "The father of {s} is {o}.",
    "mother": "The mother of {s} is {o}.",
    "spouse": "The spouse of {s} is {o}.",
    "child": "A child of {s} is {o}.",
    "sibling": "A sibling of {s} is {o}.",
    "doctoral advisor": "The doctoral advisor of {s} is {o}.",
    "date of birth": "{s} was born on {o}.",
    "date of death": "{s} died on {o}.",
    "publication date": "{s} was published on {o}.",
    "inception": "{s} was founded on {o}.",
    "place of birth": "{s} was born in {o}.",
    "place of death": "{s} died in {o}.",
    "place of burial": "{s} is buried in {o}.",
}

# docs/taxonomy.md, section 5.
EXPECTED_VERDICT = {
    "relation_reversal": "contradicted",
    "entity_swap": "contradicted",
    "wrong_relation": "contradicted",
    "fabricated_fact": "unsupported",
    "broken_chain": "contradicted",
}

DATE_RELATIONS = PERSON_DATE | WORK_DATE
# A path claim is one triple whose relation joins the hop relations (docs/taxonomy.md, 3.4).
PATH_SEP = " -> "

Skips = dict[str, int]


def group_of(relation: str) -> set[str] | None:
    """The relation group containing `relation`, or None."""
    return next((g for g in RELATION_GROUPS.values() if relation in g), None)


def surface(name: str) -> str:
    """Original spelling without a trailing disambiguator such as " (director)"."""
    return TRAILING_PAREN_RE.sub("", name) or name


def render(relation: str, subject: str, obj: str) -> str:
    """Template sentence for (subject, relation, obj); "in" instead of "on" for a bare year."""
    template = TEMPLATES[relation]
    if relation in DATE_RELATIONS and re.fullmatch(r"\d{4}", obj.strip()):
        template = template.replace(" on {o}", " in {o}")
    return template.format(s=surface(subject), o=surface(obj))


def has_relation_path(graph: dict, start: str, end: str, relations: list[str]) -> bool:
    """Whether a directed path start -> ... -> end follows exactly this relation sequence."""
    frontier = {start}
    for r in relations:
        frontier = {e["object_key"] for e in graph["edges"]
                    if e["relation"] == r and e["subject_key"] in frontier}
        if not frontier:
            return False
    return end in frontier


def classify_claim(triple: list[str], graph: dict) -> str:
    """Taxonomy rules in decision order (section 4), on normalised keys.

    Single-edge triples: rules 1, 2, 3, 5. Path claims (relation contains " -> "): rule 4.
    """
    s, r, o = normalise_entity(triple[0]), triple[1], normalise_entity(triple[2])
    if PATH_SEP in r:
        relations = [x.strip() for x in r.split(PATH_SEP.strip())]
        return "supported" if has_relation_path(graph, s, o, relations) else "broken_chain"
    edges = {(e["subject_key"], e["relation"], e["object_key"]) for e in graph["edges"]}
    if (s, r, o) in edges:
        return "supported"
    if (o, r, s) in edges:
        return "relation_reversal"
    if any(es == s and eo == o and er != r for es, er, eo in edges):
        return "wrong_relation"
    if any(er == r and ((es == s and eo != o) or (eo == o and es != s)) for es, er, eo in edges):
        return "entity_swap"
    return "fabricated_fact"


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


def _replace_sentence(row: dict, sentence: dict, new_text: str) -> str:
    """The answer with one sentence replaced."""
    answer = row.get("answer") or ""
    if sentence["text"] in answer:
        return answer.replace(sentence["text"], new_text, 1)
    return " ".join(new_text if s["index"] == sentence["index"] else s["text"]
                    for s in row["sentences"])


def _insert_after(row: dict, sentence: dict, new_text: str) -> str:
    """The answer with a new sentence inserted after `sentence`; existing text unchanged."""
    answer = row.get("answer") or ""
    pos = answer.find(sentence["text"])
    if pos >= 0:
        end = pos + len(sentence["text"])
        return answer[:end] + " " + new_text + answer[end:]
    parts = []
    for s in row["sentences"]:
        parts.append(s["text"])
        if s["index"] == sentence["index"]:
            parts.append(new_text)
    return " ".join(parts)


def _sample(row: dict, error_type: str, sentence_index: int, original_sentence: str | None,
            new_text: str, injected: str, original: list[str] | None, new: list[str]) -> dict:
    """Build one sample (seed and created are added by the main loop)."""
    return {
        "sample_id": f"{row['id']}__{error_type}",
        "source_id": row["id"],
        "question_type": row.get("type"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "error_type": error_type,
        "expected_verdict": EXPECTED_VERDICT[error_type],
        "sentence_index": sentence_index,
        "original_sentence": original_sentence,
        "new_sentence": new_text,
        "original_triple": original,
        "new_triple": new,
        "original_answer": row.get("answer") or "",
        "injected_answer": injected,
        "reference_graph": row["reference_graph"],
    }


def _edit_sample(row: dict, error_type: str, sentence: dict, new_text: str,
                 original: list[str], new: list[str]) -> dict:
    """Sample that replaces an existing sentence."""
    return _sample(row, error_type, sentence["index"], sentence["text"], new_text,
                   _replace_sentence(row, sentence, new_text), original, new)


def _direct_edges(row: dict) -> list[tuple[dict, dict]]:
    """(sentence, edge) pairs for every direct edge stated by a sentence."""
    edges = {e["index"]: e for e in row["reference_graph"]["edges"]}
    return [(s, edges[i]) for s in row["sentences"] for i in s["edges"]]


def _triple(e: dict) -> list[str]:
    return [e["subject"], e["relation"], e["object"]]


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
        out.append(_edit_sample(row, "relation_reversal", sent, new_text, _triple(e),
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
        out.append(_edit_sample(row, "entity_swap", sent, new_text, _triple(e),
                                [e["subject"], e["relation"], new_obj]))
    return out, skips


def inject_wrong_relation(row: dict, rng: random.Random,
                          pools: dict[str, list[str]]) -> tuple[list[dict], Skips]:
    """Rewrite a one-fact sentence with another relation from the same group."""
    out, skips = [], {}
    edges = {e["index"]: e for e in row["reference_graph"]["edges"]}
    with_edges = [s for s in row["sentences"] if s["edges"]]
    if not with_edges:
        _skip(skips, "no_direct_edge")
    for sent in with_edges:
        # Replacing the whole sentence must not delete other facts.
        if len(sent["edges"]) != 1 or sent["paths"]:
            _skip(skips, "multi_fact_sentence")
            continue
        e = edges[sent["edges"][0]]
        group = group_of(e["relation"])
        if group is None:
            _skip(skips, "relation_not_groupable")
            continue
        # One candidate per alternative relation; the main loop picks among the valid ones.
        for r_new in sorted(group - {e["relation"]}):
            out.append(_edit_sample(row, "wrong_relation", sent,
                                    render(r_new, e["subject"], e["object"]), _triple(e),
                                    [e["subject"], r_new, e["object"]]))
    return out, skips


def node_kinds(graph: dict) -> dict[str, str]:
    """Infer "work" or "person" per node key from its edges; conflicting or no evidence is omitted."""
    seen: dict[str, set[str]] = {}
    for e in graph["edges"]:
        r, s, o = e["relation"], e["subject_key"], e["object_key"]
        if r in FILM_PERSON or r in WORK_DATE:
            seen.setdefault(s, set()).add("work")
        if r in PERSON_PERSON:
            seen.setdefault(s, set()).add("person")
            seen.setdefault(o, set()).add("person")
        if r in FILM_PERSON:
            seen.setdefault(o, set()).add("person")
        if r in PERSON_DATE or r in PERSON_PLACE:
            seen.setdefault(s, set()).add("person")
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def inject_fabricated_fact(row: dict, rng: random.Random,
                           pools: dict[str, list[str]]) -> tuple[list[dict], Skips]:
    """Insert a new sentence stating a relation the subject has no edge for in the graph."""
    graph, out, skips = row["reference_graph"], [], {}
    graph_keys = {n["key"] for n in graph["nodes"]}
    kinds = node_kinds(graph)
    has_rel = {(e["subject_key"], e["relation"]) for e in graph["edges"]}
    done: set[str] = set()
    for sent in row["sentences"]:
        for key in sent["mentioned"]:
            if key in done:
                continue  # insert after the first sentence that mentions the node
            done.add(key)
            kind = kinds.get(key)
            if kind is None:
                _skip(skips, "unknown_kind")
                continue
            subject = _node(graph, key)["names"][0]
            for r_new in sorted(KIND_RELATIONS[kind]):
                if (key, r_new) in has_rel:
                    continue  # an existing edge would make this an entity swap
                options = [o for o in pools.get(r_new, []) if normalise_entity(o) not in graph_keys]
                if not options:
                    _skip(skips, "empty_pool")
                    continue
                o_new = rng.choice(options)
                new_text = render(r_new, subject, o_new)
                out.append(_sample(row, "fabricated_fact", sent["index"] + 1, None, new_text,
                                   _insert_after(row, sent, new_text), None,
                                   [subject, r_new, o_new]))
    if not done:
        _skip(skips, "no_mentioned_entity")
    return out, skips


def inject_broken_chain(row: dict, rng: random.Random,
                        pools: dict[str, list[str]]) -> tuple[list[dict], Skips]:
    """Replace the end of an implied path with the last-hop object of a different intermediate."""
    graph, out, skips = row["reference_graph"], [], {}
    graph_keys = {n["key"] for n in graph["nodes"]}
    edges = {e["index"]: e for e in graph["edges"]}
    path_sentences = [s for s in row["sentences"] if s["paths"] and not s["edges"]]
    if not path_sentences:
        _skip(skips, "no_path_sentence")
    for sent in path_sentences:
        for path in sent["paths"]:
            first, last = edges[path[0]], edges[path[-1]]
            r_last, x_key, o_key = last["relation"], last["subject_key"], last["object_key"]
            # a) wrong intermediate: the r_last object of another node in this graph
            wrong = [(e["object"], e["subject"]) for e in graph["edges"]
                     if e["relation"] == r_last and e["subject_key"] != x_key
                     and e["object_key"] != o_key]
            if wrong:
                new_obj, wrong_intermediate = rng.choice(wrong)
            else:
                # b) an object of the same relation from outside this graph
                options = [o for o in pools.get(r_last, []) if normalise_entity(o) not in graph_keys]
                if not options:
                    _skip(skips, "no_replacement")
                    continue
                new_obj, wrong_intermediate = rng.choice(options), None
            span = find_span(sent["text"], _node(graph, o_key))
            if span is None:
                _skip(skips, "span_not_found")
                continue
            t = sent["text"]
            new_text = t[:span[0]] + new_obj + t[span[1]:]
            relation = PATH_SEP.join(edges[i]["relation"] for i in path)
            sample = _edit_sample(row, "broken_chain", sent, new_text,
                                  [first["subject"], relation, last["object"]],
                                  [first["subject"], relation, new_obj])
            sample["path_edges"] = list(path)
            sample["wrong_intermediate"] = wrong_intermediate
            out.append(sample)
    return out, skips


Injector = Callable[[dict, random.Random, dict[str, list[str]]], tuple[list[dict], Skips]]
INJECTORS: dict[str, Injector] = {
    "relation_reversal": inject_relation_reversal,
    "entity_swap": inject_entity_swap,
    "wrong_relation": inject_wrong_relation,
    "fabricated_fact": inject_fabricated_fact,
    "broken_chain": inject_broken_chain,
}


def validation_error(sample: dict, graph: dict) -> str | None:
    """None if valid; else "invalid" (sentence unchanged) or "taxonomy_mismatch"."""
    if sample["new_sentence"] == sample["original_sentence"]:
        return "invalid"
    if classify_claim(sample["new_triple"], graph) != sample["error_type"]:
        return "taxonomy_mismatch"
    return None


def validate(sample: dict, graph: dict) -> bool:
    """A sample is valid only if its new triple classifies as its own error type."""
    return validation_error(sample, graph) is None


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
        reason = validation_error(c, row["reference_graph"])
        if reason is None:
            valid.append(c)
        else:
            _skip(skips, reason)
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

    # classify_claim
    fam = build_reference_graph([["Anne Smith", "father", "John Smith"],
                                 ["Mary Smith", "father", "Anne Smith"]])
    # wrong_relation
    film_ev = [["To Sleep with Anger", "director", "Charles Burnett"],
               ["Charles Burnett (director)", "date of birth", "April 13, 1944"]]
    wr_row = make_row(film_ev[:1], "To Sleep with Anger was directed by Charles Burnett.")
    wr, _ = inject_row(wr_row, "wrong_relation", 42, {}, {})
    country, c_skips = inject_wrong_relation(
        make_row([["Some Film", "country", "France"]], "Some Film is from France."),
        random.Random(1), {})
    multi, m_skips = inject_wrong_relation(
        make_row([["Anne Smith", "father", "John Smith"], ["Anne Smith", "mother", "Mary Jones"]],
                 "Anne Smith's parents are John Smith and Mary Jones."),
        random.Random(1), {})
    # fabricated_fact
    fab_row = make_row(film_ev, "To Sleep with Anger was directed by Charles Burnett. "
                                "Charles Burnett was born on April 13, 1944.")
    fab_pools = {"producer": ["Jane Roe"], "date of death": ["May 5, 2000"]}
    fab, _ = inject_row(fab_row, "fabricated_fact", 42, fab_pools, {})
    # taxonomy mismatch: entity swap whose new object makes the claim a reversal
    mismatch = {"error_type": "entity_swap", "original_sentence": "Anne Smith's father is John Smith.",
                "new_sentence": "Anne Smith's father is Mary Smith.",
                "new_triple": ["Anne Smith", "father", "Mary Smith"]}
    # broken_chain
    film_graph = build_reference_graph(film_ev)
    chain_row = make_row(film_ev + [["Slunce, seno, jahody", "director", "Zdeněk Troška"],
                                    ["Zdeněk Troška", "date of birth", "May 18, 1953"]],
                         "The director of To Sleep with Anger was born on April 13, 1944.")
    chain, _ = inject_row(chain_row, "broken_chain", 42, {}, {})
    no_chain, nc_skips = inject_broken_chain(wr_row, random.Random(1), {})

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
        ("classify supported",
         classify_claim(["Anne Smith", "father", "John Smith"], fam) == "supported"),
        ("classify reversal",
         classify_claim(["John Smith", "father", "Anne Smith"], fam) == "relation_reversal"),
        ("classify wrong relation",
         classify_claim(["Anne Smith", "mother", "John Smith"], fam) == "wrong_relation"),
        ("classify entity swap",
         classify_claim(["Anne Smith", "father", "Bob Lee"], fam) == "entity_swap"),
        ("classify fabricated",
         classify_claim(["Some Film", "director", "Jane Doe"], fam) == "fabricated_fact"),
        ("wrong relation same group", wr is not None
         and wr["new_triple"][1] in FILM_PERSON - {"director"}
         and "To Sleep with Anger" in wr["new_sentence"]
         and "Charles Burnett" in wr["new_sentence"]),
        ("no wrong relation for country", country == []
         and c_skips.get("relation_not_groupable") == 1),
        ("multi_fact_sentence", multi == [] and m_skips.get("multi_fact_sentence") == 1),
        ("fabricated inserts sentence", fab is not None and fab["original_sentence"] is None
         and all(s["text"] in fab["injected_answer"] for s in fab_row["sentences"])
         and fab["new_sentence"] in fab["injected_answer"]
         and len(fab["injected_answer"]) == len(fab_row["answer"]) + len(fab["new_sentence"]) + 1
         and classify_claim(fab["new_triple"], fab_row["reference_graph"]) == "fabricated_fact"),
        ("taxonomy_mismatch", validation_error(mismatch, fam) == "taxonomy_mismatch"),
        ("year-only date", render("date of birth", "Anne Smith", "1953")
         == "Anne Smith was born in 1953."),
        ("classify path supported", classify_claim(
            ["To Sleep with Anger", "director -> date of birth", "April 13, 1944"], film_graph)
         == "supported"),
        ("classify path broken", classify_claim(
            ["To Sleep with Anger", "director -> date of birth", "May 18, 1953"], film_graph)
         == "broken_chain"),
        ("broken chain wrong intermediate", chain is not None
         and chain["new_sentence"].endswith("born on May 18, 1953.")
         and chain["wrong_intermediate"] == "Zdeněk Troška"),
        ("no broken chain for direct edge", no_chain == []
         and nc_skips.get("no_path_sentence") == 1),
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
    parser.add_argument("--types", default="relation_reversal,entity_swap,wrong_relation,"
                                           "fabricated_fact,broken_chain")
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
        "broken_chain_sources": {
            "wrong_intermediate": sum(1 for s in samples if s["error_type"] == "broken_chain"
                                      and s["wrong_intermediate"] is not None),
            "pool": sum(1 for s in samples if s["error_type"] == "broken_chain"
                        and s["wrong_intermediate"] is None),
        },
        "relation_groups": {g: sorted(rels) for g, rels in RELATION_GROUPS.items()},
        "seed": args.seed,
        "created": created,
    }
    (out_dir / "inject_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Correct rows read: {len(rows)} from {mapped_path}")
    for t in types:
        print(f"  {t:<20} samples={per_type[t]}  valid candidates={candidates[t]}  "
              f"skips={dict(sorted(skips[t].items()))}")
    print(f"Symmetric: {sorted(SYMMETRIC_RELATIONS)}")
    print(f"Reversible: {sorted(REVERSIBLE_RELATIONS)}")
    print(f"Saved -> {out_dir / 'injected.jsonl'}")
    print(f"Saved -> {out_dir / 'inject_summary.json'}")


if __name__ == "__main__":
    main()
