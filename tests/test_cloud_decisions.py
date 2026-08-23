from __future__ import annotations

import ast
import re
from pathlib import Path

APP_PATH = Path(__file__).parents[1] / "cloud_demo" / "app.py"


def load_decision_namespace() -> dict:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    assignments = {
        "LANGUAGE_INSTRUCTIONS",
        "LANGUAGE_CHOICES",
        "LANGUAGE_HINTS",
        "DETECTION_BANNERS",
        "SOURCES_HEADINGS",
        "DOSE_FALLBACKS",
        "DECISION_MODES",
        "COUNTRIES",
        "PCPB_CROPS_URL",
        "NAFDAC_GREENBOOK_URL",
        "REFERENCE_CASES",
        "DOSE_PATTERN",
        "APPLICATION_LINE",
    }
    functions = {
        "detect_language",
        "reference_context",
        "label_is_usable",
        "_normalise_rate",
        "label_application_rates",
        "label_application_rate_texts",
        "sanitize_answer",
        "kenya_tomato_shortlist",
        "_source_section",
        "_matches_case",
        "cassava_purchase_card",
        "maize_fertilizer_card",
        "label_rate_card",
        "insufficient_purchase_card",
        "deterministic_purchase_answer",
        "deterministic_diagnosis_answer",
        "isolated_case_messages",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if target_names & assignments:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace


DECISIONS = load_decision_namespace()


def test_only_verified_languages_are_advertised() -> None:
    assert DECISIONS["LANGUAGE_CHOICES"] == (
        "Automatic (detect my language)",
        "English",
        "Kiswahili",
    )
    assert DECISIONS["detect_language"]("Majani ya mahindi yangu ni njano baada ya mvua") == "Kiswahili"


def test_cassava_purchase_rejects_fungicide() -> None:
    answer = DECISIONS["deterministic_purchase_answer"](
        "My cassava leaves have yellow-green mosaic and curl. Should I buy fungicide?",
        "Chemical purchase plan",
        "Kenya",
        "",
        "English",
        [],
    )
    assert "DO NOT BUY FUNGICIDE" in answer
    assert "do not treat viruses" in answer.casefold()


def test_flooded_maize_kiswahili_is_fully_localised() -> None:
    answer = DECISIONS["deterministic_purchase_answer"](
        "Mahindi yana majani ya chini ya njano baada ya mvua nyingi. Ninunue CAN?",
        "Fertilizer purchase plan",
        "Kenya",
        "",
        "Kiswahili",
        [],
    )
    assert answer.startswith("UAMUZI WA UNUNUZI")
    assert "USINUNUE CAN BADO" in answer
    assert "PURCHASE VERDICT" not in answer


def test_exact_rate_is_only_transcribed_from_complete_user_label() -> None:
    label = "Product: Example 50 WP\nActive ingredient: copper 50% WP\nApply rate: 20 g per 20 L\nPHI: 3 days"
    answer = DECISIONS["deterministic_purchase_answer"](
        "Can I use this on my tomatoes?",
        "Chemical purchase plan",
        "Kenya",
        label,
        "English",
        [],
    )
    assert "20 g per 20 L" in answer
    assert "USER-SUPPLIED LABEL" in answer
    assert "independent proof" in answer


def test_unmatched_purchase_does_not_call_the_model_or_leak_prompts() -> None:
    answer = DECISIONS["deterministic_purchase_answer"](
        "What should I buy?",
        "Chemical purchase plan",
        "Nigeria",
        "",
        "English",
        [],
    )
    assert "DO NOT BUY YET" in answer
    assert "SYSTEM_PROMPT" not in answer
    assert "NAFDAC Greenbook" in answer


def test_diagnosis_context_is_isolated_between_farmers() -> None:
    messages = DECISIONS["isolated_case_messages"](
        "System instructions for the current case",
        "New maize case",
    )
    assert messages == [
        {"role": "system", "content": "System instructions for the current case"},
        {"role": "user", "content": "New maize case\n/no_think"},
    ]
    assert "20 g per 20 L" not in repr(messages)


def test_purple_maize_diagnosis_does_not_mislabel_nitrogen() -> None:
    answer = DECISIONS["deterministic_diagnosis_answer"](
        "Young maize is purple and stunted in a cold wet field",
        ["Iowa State Extension"],
        "English",
    )
    assert answer is not None
    assert "restricted phosphorus uptake" in answer
    assert "does not prove" in answer
    assert "Do not buy phosphorus fertilizer without a soil test" in answer
    assert "nitrogen deficiency" not in answer.casefold()
