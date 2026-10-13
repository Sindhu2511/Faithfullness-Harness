"""Baseline claim extractor: one sentence = one claim (spaCy)."""
import json
from pathlib import Path

import spacy

DATA = Path("data/test_responses.json")
OUT = Path("outputs/baseline_claims.json")


def extract_claims(nlp, response_id, text):
    doc = nlp(text)
    claims = []
    for i, sent in enumerate(doc.sents, start=1):
        s = sent.text.strip()
        if s:
            claims.append({
                "claim_id": f"{response_id}-c{i}",
                "text": s,
                "source_sentence": s,
                "triple": None,  # baseline can't build triples
            })
    return claims


def main():
    nlp = spacy.load("en_core_web_sm")
    data = json.loads(DATA.read_text(encoding="utf-8"))
    results = []
    for item in data:
        claims = extract_claims(nlp, item["id"], item["response"])
        results.append({"id": item["id"], "claims": claims})
        print(f"{item['id']}: {len(claims)} claims (gold: {len(item['gold_claims'])})")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()