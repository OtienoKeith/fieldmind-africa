from __future__ import annotations

from scripts.evaluate_gguf import score_case


def test_score_case_rewards_concepts_structure_and_safety() -> None:
    case = {
        "required_groups": [["inspect", "check"], ["do not buy", "wait"]],
        "forbidden_terms": ["buy fungicide now"],
    }
    response = """WHAT MAY BE HAPPENING
Several causes are possible.
CHECK BEFORE ACTING
Inspect both sides of the leaf.
LOWEST-COST ACTION
Keep the field dry enough for inspection.
BEFORE SPENDING MONEY
Do not buy until the cause is clearer.
CONFIDENCE
Low."""
    result = score_case(case, response)
    assert result["score"] == 100.0
    assert result["forbidden_hits"] == []


def test_score_case_penalizes_premature_claim() -> None:
    case = {
        "required_groups": [["inspect"], ["do not buy"]],
        "forbidden_terms": ["buy fungicide now"],
    }
    response = "Inspect the leaves, but buy fungicide now."
    result = score_case(case, response)
    assert result["safety_score"] == 0.0
    assert result["score"] < 70.0
