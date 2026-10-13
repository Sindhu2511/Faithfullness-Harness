"""LLM claim extractor: atomic claims + triples via Ollama, with code-side checks."""
import json
import re
import time
from pathlib import Path

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

DATA = Path("data/test_responses.json")
OUT = Path("outputs/llm_claims.json")
MODEL = "qwen2.5:7b"

HEDGES = ("arguably", "probably", "possibly", "perhaps", "greatest", "most influential")


class Claim(BaseModel):
    text: str = Field(description="One atomic fact as a full sentence, pronouns replaced by names.")
    subject: str = Field(description="The entity the fact is about, e.g. Marie Curie.")
    relation: str = Field(description="Short snake_case relation, e.g. born_in, won, married, died_in_year.")
    object: str = Field(description="Only the value, e.g. Warsaw, 1879, Nobel Prize in Physics.")
    source_sentence: str = Field(description="The exact sentence from the response this fact came from.")


class ClaimList(BaseModel):
    claims: list[Claim]


PROMPT = """You split a response into atomic factual claims.

Rules:
1. One fact per claim: exactly one subject, one relation, one object.
2. Replace pronouns (he, she, they, it) with full names.
3. Skip opinions and hedged statements (arguably, probably, the greatest, the most influential). They are not claims.
4. Copy facts exactly as the response states them, even if you believe they are false. Never fix, drop, or question a fact. Your job is extraction, not fact-checking.
5. Facts inside "who" or "which" clauses are separate claims.
6. A place and a date in the same fact become two claims.
7. Triple format: relation in snake_case (born_in, won, married, died_in_year). Object is only the value, with no trailing period and no extra words.
8. If the response contains any factual statement, return at least one claim.

Example 1
Response: "Nikola Tesla, who invented the induction motor, moved to New York in 1884. He worked for Thomas Edison."
Claims:
- Nikola Tesla invented the induction motor. | Nikola Tesla | invented | induction motor
- Nikola Tesla moved to New York. | Nikola Tesla | moved_to | New York
- Nikola Tesla moved to New York in 1884. | Nikola Tesla | moved_in_year | 1884
- Nikola Tesla worked for Thomas Edison. | Nikola Tesla | worked_for | Thomas Edison

Example 2
Response: "Galileo Galilei was probably the greatest astronomer of all time. He was born in Paris."
Claims:
- Galileo Galilei was born in Paris. | Galileo Galilei | born_in | Paris
(The opinion is skipped. "Paris" is kept even though it is wrong.)

Now split this response:
Response: "{response}"
"""

RETRY_NOTE = "\nThe response above contains factual statements. Return at least one claim."


def clean(s):
    return s.strip().strip(".,;:!?\"'[]{}()=").strip()


def to_snake(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def check_triple(text, subj, rel, obj):
    """Fix what is safe to fix; flag everything else."""
    issues = []
    if rel.endswith("_year") and not re.fullmatch(r"\d{3,4}", obj):
        found = re.search(r"\b(\d{3,4})\b", text)
        if found:
            obj = found.group(1)
            issues.append("year repaired from claim text")
        else:
            issues.append("year object is not a number")
    if not obj or obj.lower() not in text.lower():
        issues.append("object not found in claim text")
    if " and " in subj:
        issues.append("compound subject")
    return obj, issues


def extract_claims(llm, response_id, text):
    result = llm.invoke(PROMPT.format(response=text))
    retried = False
    if not result.claims:
        result = llm.invoke(PROMPT.format(response=text) + RETRY_NOTE)
        retried = True
    claims = []
    for c in result.claims:
        if any(h in c.text.lower() for h in HEDGES):
            continue
        subj, rel = clean(c.subject), to_snake(c.relation)
        obj, issues = check_triple(c.text, subj, rel, clean(c.object))
        claims.append({
            "claim_id": f"{response_id}-c{len(claims) + 1}",
            "text": c.text.strip(),
            "source_sentence": c.source_sentence.strip(),
            "triple": [subj, rel, obj],
            "triple_issues": issues,
        })
    return claims, retried


def main():
    llm = ChatOllama(model=MODEL, temperature=0).with_structured_output(ClaimList)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    results = []
    for item in data:
        start = time.time()
        try:
            claims, retried = extract_claims(llm, item["id"], item["response"])
            error = None
        except Exception as e:  # bad JSON from the model, Ollama not running, etc.
            claims, retried, error = [], False, str(e)
        results.append({"id": item["id"], "claims": claims, "retried": retried, "error": error})
        print(f"{item['id']}: {len(claims)} claims (gold: {len(item['gold_claims'])}) "
              f"in {time.time() - start:.1f}s"
              + ("  [retried]" if retried else "")
              + (f"  ERROR: {error}" if error else ""))
        for c in claims:
            flag = f"  <-- {', '.join(c['triple_issues'])}" if c["triple_issues"] else ""
            print(f"    {c['text']}  {tuple(c['triple'])}{flag}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()