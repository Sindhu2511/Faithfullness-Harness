"""Phase 5: score claim extractors against hand-written gold claims."""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

DATA = Path("data/test_responses.json")
SYSTEMS = {
    "baseline": Path("outputs/baseline_claims.json"),
    "llm": Path("outputs/llm_claims.json"),
}
OUT = Path("outputs/eval_report.json")
THRESHOLD = 0.85
PRONOUNS = re.compile(r"\b(he|she|they|him|them)\b", re.IGNORECASE)


def norm(s):
    return " ".join(s.lower().strip().rstrip(".").split())


def similarity(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def match(pred, gold):
    """One-to-one greedy matching: best-scoring pairs first."""
    pairs = sorted(
        ((similarity(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)),
        reverse=True,
    )
    used_p, used_g, matched = set(), set(), []
    for score, i, j in pairs:
        if score < THRESHOLD:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, score))
    return matched


def evaluate(path, data):
    runs = json.loads(path.read_text(encoding="utf-8"))
    preds = {r["id"]: [c["text"] for c in r["claims"]] for r in runs}
    tp = n_pred = n_gold = planted_total = planted_kept = leaks = 0
    notes = []
    for item in data:
        pred, gold = preds.get(item["id"], []), item["gold_claims"]
        matched = match(pred, gold)
        matched_p = {i for i, _, _ in matched}
        matched_g = {j for _, j, _ in matched}
        tp += len(matched)
        n_pred += len(pred)
        n_gold += len(gold)
        leaks += sum(1 for p in pred if PRONOUNS.search(p))
        for err in item["planted_errors"]:
            planted_total += 1
            planted_kept += any(gold[j] == err for j in matched_g)
        for j, g in enumerate(gold):
            if j not in matched_g:
                notes.append(f"{item['id']} MISSED: {g}")
        for i, p in enumerate(pred):
            if i not in matched_p:
                notes.append(f"{item['id']} EXTRA:  {p}")
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scores = {
        "predicted": n_pred, "gold": n_gold, "matched": tp,
        "precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2),
        "planted_errors_kept": f"{planted_kept}/{planted_total}",
        "pronoun_leaks": leaks,
    }
    return scores, notes


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    report = {}
    print(f"{'system':<10}{'pred':>6}{'gold':>6}{'match':>7}{'prec':>7}{'recall':>8}{'F1':>6}{'planted':>9}{'pronoun':>9}")
    for name, path in SYSTEMS.items():
        scores, notes = evaluate(path, data)
        report[name] = {"scores": scores, "notes": notes}
        s = scores
        print(f"{name:<10}{s['predicted']:>6}{s['gold']:>6}{s['matched']:>7}{s['precision']:>7}"
              f"{s['recall']:>8}{s['f1']:>6}{s['planted_errors_kept']:>9}{s['pronoun_leaks']:>9}")
    for name in SYSTEMS:
        print(f"\n--- {name}: unmatched claims ---")
        for n in report[name]["notes"]:
            print("  " + n)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()