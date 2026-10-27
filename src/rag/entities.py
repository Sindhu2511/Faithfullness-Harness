"""Entity normalisation and the directed reference graph built from gold evidences.

The reference graph G for a question is its gold evidence triples (docs/decisions.md, D1).
These helpers are shared by sentence mapping (B5) and, later, the verifier.
"""
import re
import unicodedata

TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _normalise_text(text: str) -> str:
    """NFKD + strip accents, lowercase, punctuation to spaces, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())


def normalise_entity(name: str) -> str:
    """Normalised entity key: like _normalise_text, after dropping a trailing "(...)"."""
    return _normalise_text(TRAILING_PAREN_RE.sub("", name or ""))


def build_reference_graph(evidences: list[list[str]]) -> dict:
    """Directed graph from gold evidence triples, with alias spellings merged by key."""
    names: dict[str, list[str]] = {}
    edges = []
    for i, (subj, rel, obj) in enumerate(evidences):
        s_key, o_key = normalise_entity(subj), normalise_entity(obj)
        for key, name in ((s_key, subj), (o_key, obj)):
            spellings = names.setdefault(key, [])
            if name not in spellings:
                spellings.append(name)
        edges.append({"index": i, "subject": subj, "relation": rel, "object": obj,
                      "subject_key": s_key, "object_key": o_key})
    return {
        "nodes": [{"key": k, "names": v} for k, v in names.items()],
        "edges": edges,
        "alias_merges": [k for k, v in names.items() if len(v) > 1],
    }


def mentions(text: str, node: dict) -> bool:
    """Whether any spelling of the node appears in text (whole words, or all tokens any order)."""
    norm_text = _normalise_text(text)
    text_tokens = set(norm_text.split())
    for name in node["names"]:
        norm_name = normalise_entity(name)
        if not norm_name:
            continue
        if f" {norm_name} " in f" {norm_text} ":
            return True
        name_tokens = norm_name.split()
        # Multi-word values such as dates: "april 13 1944" matches "13 april 1944".
        if len(name_tokens) > 1 and all(t in text_tokens for t in name_tokens):
            return True
    return False


def find_paths(graph: dict, start_key: str, end_key: str, max_hops: int) -> list[list[int]]:
    """All directed paths of 2..max_hops edges from start to end (edge indexes, no repeated node)."""
    out_edges: dict[str, list[dict]] = {}
    for e in graph["edges"]:
        out_edges.setdefault(e["subject_key"], []).append(e)

    paths: list[list[int]] = []

    def walk(node: str, visited: set[str], path: list[int]) -> None:
        if len(path) >= max_hops:
            return
        for e in out_edges.get(node, []):
            nxt = e["object_key"]
            if nxt in visited:
                continue
            if nxt == end_key:
                if len(path) + 1 >= 2:
                    paths.append(path + [e["index"]])
                continue
            walk(nxt, visited | {nxt}, path + [e["index"]])

    if start_key != end_key:
        walk(start_key, {start_key}, [])
    return paths
