from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path

import gradio as gr
import requests
import spaces
from huggingface_hub import hf_hub_download

MODEL_REPO = os.environ.get("MODEL_REPO", "otieno28/fieldmind-africa-1.7b-gguf")
MODEL_FILE = os.environ.get("MODEL_FILE", "FieldMind-Africa-1.7B-Q5_K_M.gguf")
MODEL_LABEL = os.environ.get("MODEL_LABEL", "FieldMind Africa 1.7B Q5_K_M — trained submission model")
LLAMA_RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b10593/llama-b10593-bin-ubuntu-x64.tar.gz"
LLAMA_RUNTIME_SHA256 = "fb3479dcb6b8ced8d785585af991b9ffa6ce605b51fc06229c1770544922a0db"
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8081"))
THREADS = int(os.environ.get("LLAMA_THREADS", "2"))

LANGUAGE_INSTRUCTIONS = {
    "English": "Reply fully in clear, simple English.",
    "Kiswahili": "Jibu kikamilifu kwa Kiswahili rahisi na wazi.",
}

LANGUAGE_CHOICES = ("Automatic (detect my language)", *LANGUAGE_INSTRUCTIONS)

LANGUAGE_HINTS = {
    "English": {"my", "the", "what", "should", "leaves", "after", "buy", "plant", "crop", "rain"},
    "Kiswahili": {"mahindi", "majani", "nyanya", "mihogo", "muhogo", "mvua", "ninunue", "dawa", "shamba", "baada", "njano", "sasa", "yangu", "yana"},
}

PROGRESS_MESSAGES = {
    "English": "🌱 Checking the safest, lowest-cost next step…",
    "Kiswahili": "🌱 Ninakagua hatua salama na yenye gharama ya chini…",
}

DETECTION_BANNERS = {
    "English": "Detected language: English",
    "Kiswahili": "Lugha iliyotambuliwa: Kiswahili",
}

SOURCES_HEADINGS = {
    "English": "SOURCES USED",
    "Kiswahili": "VYANZO VILIVYOTUMIKA",
}

DOSE_FALLBACKS = {
    "English": "DOSE: Paste the exact registered product label to receive its verified rate; never transfer a dose between formulations.",
    "Kiswahili": "KIPIMO: Bandika lebo kamili ya bidhaa iliyosajiliwa ili kupata kipimo kilichothibitishwa; usihamishe kipimo kati ya formulations.",
}

DECISION_MODES = (
    "Diagnose and decide",
    "Chemical purchase plan",
    "Fertilizer purchase plan",
)

COUNTRIES = ("Kenya", "Nigeria", "Other African country")

PCPB_CROPS_URL = "https://www.pcpb.go.ke/crops/"
NAFDAC_GREENBOOK_URL = "https://greenbook.nafdac.gov.ng/"

KENYA_TOMATO_BLIGHT_NOTE = """The Kenya PCPB crop registry lists these tomato options for the named targets:
- AMIDIL 68WG — metalaxyl-M 40 g/kg + mancozeb 640 g/kg — registered for early and late blight; tomato PHI 3 days; REI 12 hours.
- BLUE COP 47 WP — copper oxychloride 47% — registered for early blight; tomato PHI 3 days.
These are conditional options, not proof of diagnosis. The g/kg and percentage numbers above are ingredient concentrations, NOT application doses. The verdict must be SHORTLIST until field checks support the target. Do not call a product safe. Do not state an application rate unless it is copied from the user's exact registered product label. Never transfer a dose between products or formulations."""

REFERENCE_CASES = (
    {
        "crop": ("cassava", "mihogo", "muhogo"),
        "clues": ("mosaic", "curl", "yellow-green", "yellow green"),
        "note": (
            "Cassava mosaic-like chlorosis and leaf distortion can indicate cassava mosaic disease, but text alone cannot confirm it. "
            "Fungicide does not treat a viral disease. IITA guidance emphasizes healthy planting cuttings and tolerant varieties."
        ),
        "source": "[IITA cassava disease IPM guide](https://www.iita.org/wp-content/uploads/2016/06/Disease_control_in_cassava_farms_IPM_field_guide_for_extension_agents-1.pdf)",
    },
    {
        "crop": ("maize", "corn", "mahindi"),
        "clues": ("yellow", "njano"),
        "note": (
            "Saturated soil can restrict maize root growth and nutrient uptake and can increase nitrogen loss. "
            "Nitrogen-deficiency yellowing commonly begins on lower leaves along the midrib from the tip, but fertilizer should follow field and soil checks."
        ),
        "source": "[University of Minnesota Extension: flooded corn](https://extension.umn.edu/growing-corn/flooded-corn)",
    },
    {
        "crop": ("tomato", "nyanya"),
        "clues": ("spot", "brown", "black", "doa"),
        "note": (
            "Tomato leaf spots can come from several fungal or bacterial diseases that can look similar. "
            "Spot size, rings, water-soaking, leaf position, fruit symptoms, weather, and spread help separate them; exact chemical control requires diagnosis and the local product label."
        ),
        "source": "[University of Minnesota Extension: tomato leaf spots](https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/tomato/leavesspots.html)",
    },
    {
        "crop": ("maize", "corn", "mahindi"),
        "clues": ("purple", "purplish", "zambarau"),
        "note": (
            "Purple, stunted young maize in cold or wet soil can reflect temporarily restricted phosphorus uptake or restricted roots; "
            "it does not by itself prove the soil lacks phosphorus. Check roots, compaction, drainage, soil temperature, and a soil test before buying fertilizer."
        ),
        "source": "[Iowa State University Extension: plant color, management and weather](https://crops.extension.iastate.edu/encyclopedia/plant-color-differences-based-management-and-weather)",
    },
)


@spaces.GPU(duration=1)
def zero_gpu_healthcheck() -> str:
    """Register a tiny ZeroGPU endpoint; model inference itself stays CPU-only."""
    return "FieldMind is ready"

SYSTEM_PROMPT = """You are FieldMind Africa, an offline agricultural purchase-decision assistant for extension officers, cooperatives, community centres, and input shops serving smallholder farmers.

Your job is to reach a useful purchase verdict, not merely list possible causes. Separate observations from possible causes and do not pretend a text description proves one diagnosis. When verified registry evidence matches, name the relevant active ingredient and registered product options, state the target each option is registered for, and say whether to BUY, SHORTLIST, or DO NOT BUY. When evidence points to a non-chemical problem, clearly say DO NOT BUY the chemical and give the better action.

An exact dose is product-, formulation-, crop-, target-, and country-specific. Give an exact rate only when the user has supplied the exact registered label text. Copy the rate from that label; never invent it, infer it from an active ingredient, or transfer it between products. Mention label PHI, REI, PPE, and maximum applications when available. For urgent poisoning, animal distress, or rapidly spreading serious disease, direct the user to a trained local professional promptly.

For purchase modes use these headings in the user's language: PURCHASE VERDICT; LIKELY TARGET; REGISTERED OPTION; DOSE; CHECK BEFORE USE; SAFETY; CONFIDENCE. For diagnosis mode use: WHAT MAY BE HAPPENING; CHECK BEFORE ACTING; LOWEST-COST ACTION; BEFORE SPENDING MONEY; CONFIDENCE. Keep the answer practical and under 180 words."""


def detect_language(message: str) -> str:
    """Detect the two verified writing languages without a paid service or network call."""
    lowered = message.casefold()
    tokens = set(re.findall(r"[^\W\d_]+", lowered, flags=re.UNICODE))
    scores = {language: len(tokens & hints) for language, hints in LANGUAGE_HINTS.items()}
    detected, score = max(scores.items(), key=lambda item: item[1])
    return detected if score else "English"


def reference_context(message: str) -> tuple[str, list[str]]:
    text = message.casefold()
    notes: list[str] = []
    sources: list[str] = []
    for case in REFERENCE_CASES:
        if any(word in text for word in case["crop"]) and any(word in text for word in case["clues"]):
            notes.append(case["note"])
            sources.append(case["source"])
    return "\n".join(notes), sources


def purchase_context(message: str, decision_mode: str, country: str) -> tuple[str, list[str]]:
    if decision_mode != "Chemical purchase plan":
        return "", []
    text = message.casefold()
    is_tomato_spot_case = any(word in text for word in ("tomato", "nyanya")) and any(
        word in text for word in ("spot", "brown", "black", "doa", "blight", "ring", "water-soaked", "water soaked")
    )
    if country == "Kenya" and is_tomato_spot_case:
        return KENYA_TOMATO_BLIGHT_NOTE, [f"[Kenya PCPB registered crop products]({PCPB_CROPS_URL})"]
    if country == "Nigeria":
        return (
            "Use Nigeria's NAFDAC Greenbook to verify the exact product and registration. No case-specific registered product was matched, "
            "so request the crop, target problem, product name, active ingredient, formulation, and label text before naming a purchase option."
        ), [f"[Nigeria NAFDAC Greenbook]({NAFDAC_GREENBOOK_URL})"]
    return (
        "No country-specific registered product was matched. Request the crop, target problem, country, product name, formulation, and exact label text."
    ), []


DOSE_PATTERN = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*(?:%|ml|millilit(?:er|re)s?|g|kg|lit(?:er|re)s?|l)"
    r"(?:\s*(?:/|per)\s*(?:\d+(?:\.\d+)?\s*)?(?:l|lit(?:er|re)s?|ha|hectare|kg))?\b"
)

APPLICATION_LINE = re.compile(
    r"(?i)\b(dose|rate|mix|apply|spray|dilut\w*|tank|per\s+(?:ha|hectare)|"
    r"dozi|kipimo|kiwango|changanya|nyunyiza|tumia|taux|mélang\w*|pulvéris\w*|appliqu\w*|"
    r"adadin|fesa|haɗa|ìwọ̀n|iwọn|dapọ|ọnụọgụgụ)\b"
)


def label_is_usable(label_text: str) -> bool:
    text = label_text.strip()
    has_rate = bool(DOSE_PATTERN.search(text))
    has_identity = bool(re.search(r"(?i)(product|active ingredient|formulation|wp\b|wg\b|ec\b|sc\b|sl\b)", text))
    return len(text) >= 30 and has_rate and has_identity


def _normalise_rate(value: str) -> str:
    compact = re.sub(r"\s+", "", value.casefold()).replace("litres", "l").replace("liters", "l")
    return compact.replace("per", "/")


def label_application_rates(label_text: str) -> set[str]:
    if not label_is_usable(label_text):
        return set()
    rates: set[str] = set()
    for line in label_text.splitlines():
        for match in DOSE_PATTERN.finditer(line):
            preceding_text = line[max(0, match.start() - 55):match.start()]
            if APPLICATION_LINE.search(preceding_text):
                rates.add(_normalise_rate(match.group(0)))
    return rates


def label_application_rate_texts(label_text: str) -> list[str]:
    """Return verbatim application-rate fragments found in the user-supplied label."""
    if not label_is_usable(label_text):
        return []
    rate_texts: list[str] = []
    for line in label_text.splitlines():
        for match in DOSE_PATTERN.finditer(line):
            preceding_text = line[max(0, match.start() - 55):match.start()]
            value = match.group(0).strip()
            if APPLICATION_LINE.search(preceding_text) and value not in rate_texts:
                rate_texts.append(value)
    return rate_texts


def sanitize_answer(text: str, label_text: str = "", trusted_context: str = "", language: str = "English") -> str:
    """Allow named options, but never allow an application dose absent from the supplied label's rate directions."""
    allowed_rates = label_application_rates(label_text)
    safe_lines: list[str] = []
    replacement_used = False
    for line in text.splitlines():
        line_rates = [_normalise_rate(match.group(0)) for match in DOSE_PATTERN.finditer(line)]
        if APPLICATION_LINE.search(line) and line_rates and any(rate not in allowed_rates for rate in line_rates):
            if not replacement_used:
                safe_lines.append(
                    DOSE_FALLBACKS.get(language, DOSE_FALLBACKS["English"])
                )
                replacement_used = True
        else:
            safe_lines.append(line)
    return "\n".join(safe_lines)


def kenya_tomato_shortlist(sources: list[str], language: str) -> str:
    source_list = "\n".join(f"- {source}" for source in sources)
    if language == "Kiswahili":
        return f"""UAMUZI WA UNUNUZI: WEKA KWENYE ORODHA FUPI — usinunue hadi ukaguzi ulio hapa chini uunge mkono early blight.

KISABABISHI KINACHOWEZEKANA: Early blight inaendana na madoa yenye duara yanayoanzia kwenye majani ya chini, lakini maelezo ya maandishi pekee hayathibitishi ugonjwa.

CHAGUO ZILIZOSAJILIWA:
- AMIDIL 68WG — metalaxyl-M + mancozeb; imesajiliwa na PCPB kwa early/late blight ya nyanya.
- BLUE COP 47 WP — copper oxychloride; imesajiliwa na PCPB kwa early blight ya nyanya.

KIPIMO: Bandika maandishi ya lebo ya bidhaa husika. Kiwango cha kiambato si kipimo cha kunyunyizia, na vipimo havihamishwi kati ya formulations.

KABLA YA KUTUMIA: Thibitisha madoa yenye duara, kuanzia majani ya chini, na kutokuwepo kwa madoa yenye maji au pembe.

USALAMA: Fuata PPE, PHI, REI, idadi ya matumizi na maelekezo ya utupaji yaliyo kwenye lebo.

UHAKIKA: Wa kati.

VYANZO VILIVYOTUMIKA
{source_list}"""
    return f"""PURCHASE VERDICT: SHORTLIST — do not buy until the checks below support early blight.

LIKELY TARGET: Early blight is consistent with concentric rings beginning on lower leaves, but text alone does not confirm it.

REGISTERED OPTIONS:
- AMIDIL 68WG — metalaxyl-M + mancozeb; PCPB-listed for tomato early/late blight.
- BLUE COP 47 WP — copper oxychloride; PCPB-listed for tomato early blight.

DOSE: Paste the exact product label. The ingredient concentration is not the spray dose, and rates cannot be transferred between formulations.

CHECK BEFORE USE: Confirm concentric target-like rings, lower-leaf start, and absence of water-soaked/angular spots. If uncertain, ask an extension officer.

SAFETY: Follow the chosen label's PPE, PHI, REI, maximum applications, and disposal directions.

CONFIDENCE: Medium.

SOURCES USED
{source_list}"""


def _source_section(sources: list[str], language: str) -> str:
    if not sources:
        return ""
    heading = SOURCES_HEADINGS.get(language, SOURCES_HEADINGS["English"])
    return f"\n\n{heading}\n" + "\n".join(f"- {source}" for source in sources)


def _matches_case(message: str, case_index: int) -> bool:
    text = message.casefold()
    case = REFERENCE_CASES[case_index]
    return any(word in text for word in case["crop"]) and any(word in text for word in case["clues"])


def cassava_purchase_card(sources: list[str], language: str) -> str:
    if language == "Kiswahili":
        answer = """UAMUZI WA UNUNUZI: USINUNUE DAWA YA KUVU.

KISABABISHI KINACHOWEZEKANA: Dalili za mosaic ya njano-kijani na majani kupinda zinaweza kuendana na ugonjwa wa cassava mosaic, lakini maelezo pekee hayathibitishi ugonjwa.

KWA NINI: Dawa ya kuvu haitibu virusi. Kununua fungicide hapa kunaweza kupoteza pesa bila kutatua chanzo.

KABLA YA KUCHUKUA HATUA: Angalia kama dalili ziko kwenye mimea mingi, kama vipando vilikuwa na dalili, na kama wadudu weupe wapo chini ya majani. Pata uthibitisho wa afisa ugani ikiwa inaenea haraka.

HATUA YA GHARAMA YA CHINI: Tenga mimea yenye dalili kali, usitumie vipando vyake, na tumia vipando safi na aina zinazostahimili ugonjwa katika msimu ujao.

UHAKIKA: Wa kati."""
    else:
        answer = """PURCHASE VERDICT: DO NOT BUY FUNGICIDE.

LIKELY TARGET: Yellow-green mosaic and curled cassava leaves can fit cassava mosaic disease, but a text description cannot confirm it.

WHY: Fungicides do not treat viruses. Buying one for this pattern could spend the input budget without treating the cause.

CHECK BEFORE ACTING: Check whether many plants are affected, whether the planting cuttings had symptoms, and whether whiteflies are present under leaves. Ask an extension officer if spread is rapid.

LOWEST-COST ACTION: Separate severely affected plants, do not reuse their cuttings, and use clean planting material and tolerant varieties next season.

CONFIDENCE: Medium."""
    return answer + _source_section(sources, language)


def maize_fertilizer_card(sources: list[str], language: str) -> str:
    if language == "Kiswahili":
        answer = """UAMUZI WA UNUNUZI: SUBIRI — USINUNUE CAN BADO.

KINACHOWEZA KUTOKEA: Udongo uliojaa maji unaweza kuzuia mizizi na ufyonzaji wa virutubisho, na pia kuongeza upotevu wa nitrojeni. Njano inayoanzia kwenye ncha ya jani la chini kuelekea katikati inaweza kuonyesha upungufu wa nitrojeni, lakini mvua pekee haitoshi kuthibitisha hilo.

KABLA YA KUNUNUA: Ondoa maji yaliyotuama; kagua mizizi, sehemu ya shamba iliyoathirika, na umbo la njano kwenye majani ya chini. Linganisha sehemu yenye maji na sehemu kavu, kisha tumia kipimo cha udongo au ushauri wa afisa ugani.

KIPIMO: Hakuna kipimo salama kinachoweza kutolewa bila matokeo ya shamba/udongo na lebo kamili ya mbolea husika.

HATUA YA GHARAMA YA CHINI: Rekebisha mifereji kwanza na tathmini mimea baada ya udongo kuruhusu hewa kuingia.

UHAKIKA: Wa kati."""
    else:
        answer = """PURCHASE VERDICT: WAIT — DO NOT BUY CAN YET.

WHAT MAY BE HAPPENING: Waterlogged soil can restrict roots and nutrient uptake and increase nitrogen loss. Yellowing that starts at the tip of older lower leaves along the midrib can support nitrogen deficiency, but heavy rain alone does not prove it.

CHECK BEFORE BUYING: Drain standing water; inspect roots, the field pattern, and the shape of yellowing on lower leaves. Compare wet and better-drained areas, then use a soil check or local extension advice.

DOSE: No safe rate can be given without the field/soil result and the exact fertilizer label.

LOWEST-COST ACTION: Correct drainage first and reassess plants after the soil is aerated.

CONFIDENCE: Medium."""
    return answer + _source_section(sources, language)


def label_rate_card(label_text: str, sources: list[str], language: str) -> str:
    rates = label_application_rate_texts(label_text)
    displayed_rates = ", ".join(rates)
    if language == "Kiswahili":
        answer = f"""UAMUZI WA UNUNUZI: THIBITISHA KABLA YA KUTUMIA.

KIPIMO KILICHONAKILIWA KUTOKA LEBO YA MTUMIAJI: {displayed_rates}

HILI LINAMAANISHA NINI: Hii ni nakala ya kipimo kilicho kwenye maandishi uliyobandika; si uthibitisho huru wa usajili au kuwa bidhaa inafaa kwa tatizo hili.

KABLA YA KUTUMIA: Hakikisha lebo ni ya bidhaa, formulation, zao, tatizo na nchi hiyo hiyo. Fuata PPE, PHI, REI, idadi ya juu ya matumizi na maelekezo ya kuchanganya yaliyo kwenye lebo. Usibadilishe kipimo kati ya formulations.

UHAKIKA: Wa juu kuhusu kunakili kipimo; wa chini kuhusu kama bidhaa inafaa bila lebo kamili na utambuzi."""
    else:
        answer = f"""PURCHASE VERDICT: VERIFY BEFORE USE.

RATE TRANSCRIBED FROM THE USER-SUPPLIED LABEL: {displayed_rates}

WHAT THIS MEANS: This is a transcription of the rate in the text you pasted; it is not independent proof of registration or that the product fits this problem.

CHECK BEFORE USE: Confirm the label is for the exact product, formulation, crop, target and country. Follow its PPE, PHI, REI, maximum applications and mixing directions. Never transfer the rate between formulations.

CONFIDENCE: High for the transcription; low for product suitability without the full label and diagnosis."""
    return answer + _source_section(sources, language)


def insufficient_purchase_card(country: str, sources: list[str], language: str) -> str:
    if language == "Kiswahili":
        answer = """UAMUZI WA UNUNUZI: USINUNUE BADO — ushahidi hautoshi kuchagua bidhaa kwa usalama.

TAARIFA INAYOHITAJIKA: Taja zao na aina yake, umri wa mmea, dalili halisi, kama majani ya zamani au mapya yameathirika, sehemu ya shamba iliyoathirika, hali ya hewa na mifereji, na nchi.

KWA KEMIKALI AU MBOLEA MAALUM: Bandika jina la bidhaa, kiambato/formulation, zao na tatizo kwenye lebo, kipimo, PHI, REI na maelekezo ya usalama.

HATUA YA GHARAMA YA CHINI: Piga picha za karibu na za shamba lote, linganisha mimea iliyoathirika na isiyoathirika, na uombe uthibitisho wa afisa ugani au rejista ya nchi kabla ya kutumia pesa.

UHAKIKA: Wa juu kwamba maelezo zaidi yanahitajika."""
    else:
        answer = """PURCHASE VERDICT: DO NOT BUY YET — there is not enough verified evidence to choose a product safely.

INFORMATION NEEDED: Give the crop and variety, crop age, exact symptoms, whether old or new leaves are affected, the field pattern, recent weather and drainage, and the country.

FOR A SPECIFIC CHEMICAL OR FERTILIZER: Paste the product name, active ingredient/formulation, label crop and target, rate, PHI, REI and safety directions.

LOWEST-COST ACTION: Take close and whole-field photos, compare affected and healthy plants, and get an extension or country-registry check before spending money.

CONFIDENCE: High that more evidence is required."""
    if country == "Nigeria":
        registry = f"[Nigeria NAFDAC Greenbook]({NAFDAC_GREENBOOK_URL})"
        if registry not in sources:
            sources.append(registry)
    return answer + _source_section(sources, language)


def deterministic_purchase_answer(
    message: str,
    decision_mode: str,
    country: str,
    product_label: str,
    language: str,
    sources: list[str],
) -> str:
    if product_label.strip() and label_is_usable(product_label) and label_application_rate_texts(product_label):
        return label_rate_card(product_label, sources, language)
    if decision_mode == "Chemical purchase plan" and country == "Kenya" and _matches_case(message, 2):
        return kenya_tomato_shortlist(sources, language)
    if decision_mode == "Chemical purchase plan" and _matches_case(message, 0):
        return cassava_purchase_card(sources, language)
    if decision_mode == "Fertilizer purchase plan" and _matches_case(message, 1):
        return maize_fertilizer_card(sources, language)
    return insufficient_purchase_card(country, sources, language)


def deterministic_diagnosis_answer(message: str, sources: list[str], language: str) -> str | None:
    """Return evidence-bounded cards for reference cases; leave unmatched cases to the GGUF."""
    text = message.casefold()
    is_cassava = _matches_case(message, 0)
    is_flooded_maize = _matches_case(message, 1) and any(
        word in text for word in ("rain", "mvua", "wet", "waterlog", "flood", "drain")
    )
    is_tomato_spot = _matches_case(message, 2)
    is_purple_maize = _matches_case(message, 3)
    if language == "Kiswahili":
        if is_cassava:
            answer = """KINACHOWEZA KUTOKEA: Mosaic ya njano-kijani na majani ya muhogo kupinda inaweza kuendana na cassava mosaic, lakini maandishi pekee hayathibitishi ugonjwa.

KAGUA KABLA YA HATUA: Angalia mpangilio wa dalili shambani, vipando vilivyotumika na wadudu weupe chini ya majani. Pata uthibitisho wa afisa ugani ikiwa inaenea haraka.

HATUA YA GHARAMA YA CHINI: Tenga mimea iliyoathirika sana na usitumie vipando vyake.

KABLA YA KUTUMIA PESA: Usinunue fungicide; haitibu virusi.

UHAKIKA: Wa kati."""
        elif is_flooded_maize:
            answer = """KINACHOWEZA KUTOKEA: Maji mengi yanaweza kuzuia mizizi na ufyonzaji wa virutubisho; njano ya jani la chini inaweza pia kuendana na upungufu wa nitrojeni, lakini dalili pekee hazithibitishi hilo.

KAGUA KABLA YA HATUA: Ondoa maji yaliyotuama, kagua mizizi na umbo la njano, na linganisha sehemu yenye maji na sehemu kavu.

HATUA YA GHARAMA YA CHINI: Rekebisha mifereji, subiri udongo upate hewa, kisha tumia kipimo cha udongo au afisa ugani.

KABLA YA KUTUMIA PESA: Usinunue CAN bado.

UHAKIKA: Wa kati."""
        elif is_tomato_spot:
            answer = """KINACHOWEZA KUTOKEA: Madoa ya nyanya yanaweza kusababishwa na magonjwa tofauti ya kuvu au bakteria yanayofanana.

KAGUA KABLA YA HATUA: Angalia duara kwenye doa, kingo zenye maji, jani la zamani au jipya, matunda, hali ya hewa na kasi ya kuenea.

HATUA YA GHARAMA YA CHINI: Ondoa majani yaliyoathirika sana, epuka kumwagilia majani na pata uthibitisho wa afisa ugani.

KABLA YA KUTUMIA PESA: Usichague dawa au kipimo kabla ya kuthibitisha tatizo na lebo ya nchi yako.

UHAKIKA: Wa kati."""
        elif is_purple_maize:
            answer = """KINACHOWEZA KUTOKEA: Mahindi machanga ya zambarau kwenye udongo baridi au wenye maji yanaweza kuwa na ufyonzaji mdogo wa fosforasi kwa muda au mizizi iliyozuiwa; si uthibitisho kwamba fosforasi ya udongo ni kidogo.

KAGUA KABLA YA HATUA: Chimba mimea michache; kagua mizizi, mgandamizo, mifereji, joto la udongo na matokeo ya kipimo cha udongo.

HATUA YA GHARAMA YA CHINI: Rekebisha maji au mgandamizo na uangalie ukuaji mpya udongo unapopata joto.

KABLA YA KUTUMIA PESA: Usinunue mbolea ya fosforasi bila kipimo cha udongo.

UHAKIKA: Wa kati."""
        else:
            return None
    else:
        if is_cassava:
            answer = """WHAT MAY BE HAPPENING: Yellow-green mosaic and curled cassava leaves can fit cassava mosaic disease, but text alone cannot confirm it.

CHECK BEFORE ACTING: Check the field pattern, planting cuttings and whiteflies under leaves. Ask an extension officer if spread is rapid.

LOWEST-COST ACTION: Separate severely affected plants and do not reuse their cuttings.

BEFORE SPENDING MONEY: Do not buy fungicide; it does not treat a virus.

CONFIDENCE: Medium."""
        elif is_flooded_maize:
            answer = """WHAT MAY BE HAPPENING: Saturated soil can restrict roots and nutrient uptake. Lower-leaf yellowing can also fit nitrogen shortage, but symptoms alone do not prove it.

CHECK BEFORE ACTING: Drain standing water, inspect roots and the yellowing pattern, and compare wet with better-drained areas.

LOWEST-COST ACTION: Correct drainage, let the soil aerate, then use a soil check or extension advice.

BEFORE SPENDING MONEY: Do not buy CAN yet.

CONFIDENCE: Medium."""
        elif is_tomato_spot:
            answer = """WHAT MAY BE HAPPENING: Several fungal and bacterial tomato leaf spots can look alike.

CHECK BEFORE ACTING: Inspect rings, water-soaked or angular edges, old versus new leaves, fruit symptoms, weather and spread.

LOWEST-COST ACTION: Remove badly affected leaves, avoid wetting foliage and get an extension check.

BEFORE SPENDING MONEY: Do not choose a chemical or dose until the target and local product label are verified.

CONFIDENCE: Medium."""
        elif is_purple_maize:
            answer = """WHAT MAY BE HAPPENING: Purple, stunted young maize in cold or wet soil can reflect temporarily restricted phosphorus uptake or restricted roots; it does not prove the soil lacks phosphorus.

CHECK BEFORE ACTING: Dig several plants and inspect roots, compaction, drainage, soil temperature and a soil test.

LOWEST-COST ACTION: Correct water or compaction problems and watch new growth as soil warms.

BEFORE SPENDING MONEY: Do not buy phosphorus fertilizer without a soil test.

CONFIDENCE: Medium."""
        else:
            return None
    return answer + _source_section(sources, language)


def isolated_case_messages(grounded_prompt: str, message: str) -> list[dict[str, str]]:
    """Build a one-case chat so an earlier farmer's product context cannot leak."""
    return [
        {"role": "system", "content": grounded_prompt},
        {"role": "user", "content": f"{message}\n/no_think"},
    ]


def wait_for_backend(process: subprocess.Popen, timeout: int = 240) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            response = requests.get(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)
    raise TimeoutError("Model backend did not become ready")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Unsafe runtime archive member: {member.name}")
    archive.extractall(destination)


def ensure_llama_server() -> Path:
    explicit = os.environ.get("LLAMA_SERVER")
    if explicit:
        return Path(explicit)
    runtime_dir = Path.home() / ".cache/fieldmind/llama-b10593"
    existing = next(runtime_dir.rglob("llama-server"), None) if runtime_dir.exists() else None
    if existing:
        return existing
    runtime_dir.mkdir(parents=True, exist_ok=True)
    archive_path = runtime_dir / "llama-runtime.tar.gz"
    with requests.get(LLAMA_RUNTIME_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                handle.write(chunk)
    if sha256(archive_path) != LLAMA_RUNTIME_SHA256:
        raise RuntimeError("llama.cpp runtime checksum mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, runtime_dir)
    server = next(runtime_dir.rglob("llama-server"), None)
    if server is None:
        raise RuntimeError("llama-server missing from verified runtime archive")
    server.chmod(0o755)
    return server


def start_backend() -> subprocess.Popen:
    model_path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))
    llama_server = ensure_llama_server()
    command = [
        str(llama_server),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(BACKEND_PORT),
        "-t", str(THREADS),
        "-c", "2048",
        "-ngl", "0",
        "--jinja",
    ]
    process = subprocess.Popen(command)
    wait_for_backend(process)
    return process


BACKEND = start_backend()


def answer(
    message: str,
    history: list[dict],
    language: str,
    decision_mode: str,
    country: str,
    product_label: str,
):
    evidence, sources = reference_context(message)
    purchase_evidence, purchase_sources = purchase_context(message, decision_mode, country)
    sources.extend(source for source in purchase_sources if source not in sources)
    resolved_language = detect_language(message) if language == "Automatic (detect my language)" else language
    detection_prefix = (
        DETECTION_BANNERS.get(resolved_language, DETECTION_BANNERS["English"]) + "\n\n"
        if language == "Automatic (detect my language)" else ""
    )
    language_rule = LANGUAGE_INSTRUCTIONS.get(resolved_language, LANGUAGE_INSTRUCTIONS["English"])
    language_rule += " Every heading and sentence must use this language; keep only registered product and active-ingredient names unchanged."
    if decision_mode in {"Chemical purchase plan", "Fertilizer purchase plan"}:
        yield detection_prefix + deterministic_purchase_answer(
            message,
            decision_mode,
            country,
            product_label,
            resolved_language,
            sources,
        )
        return
    diagnosis_card = deterministic_diagnosis_answer(message, sources, resolved_language)
    if diagnosis_card:
        yield detection_prefix + diagnosis_card
        return
    grounded_prompt = SYSTEM_PROMPT + f"\n\nLANGUAGE: {language_rule}\nMODE: {decision_mode}\nCOUNTRY: {country}"
    if evidence:
        grounded_prompt += (
            "\n\nVERIFIED REFERENCE NOTES FOR THIS CASE:\n"
            + evidence
            + "\nUse these notes as limited evidence. Do not add a diagnosis that the notes do not support."
        )
    else:
        grounded_prompt += (
            "\n\nNo verified disease reference note matched this case. Be explicit about the evidence gap and request the decisive field checks."
        )
    if purchase_evidence:
        grounded_prompt += "\n\nVERIFIED PURCHASE CONTEXT:\n" + purchase_evidence
    if product_label.strip():
        if label_is_usable(product_label):
            grounded_prompt += (
                "\n\nUSER-SUPPLIED PRODUCT LABEL TEXT:\n"
                + product_label.strip()
                + "\nThe label has identifiable dose information. You may copy only rates present in this text. Make clear that the label text was user-supplied."
            )
        else:
            grounded_prompt += (
                "\n\nThe user supplied incomplete label text. Do not give a dose. Ask for the product name, active ingredient, formulation, and complete rate directions."
            )
    elif decision_mode in {"Chemical purchase plan", "Fertilizer purchase plan"}:
        grounded_prompt += "\n\nNo exact product label was supplied. Name verified options when supported, but state that the exact dose requires the exact registered label."
    # Treat every field case as an isolated decision. Carrying earlier chat turns
    # can leak a previous product, dose, crop, or country into a new farmer's
    # case, which is unsafe and also makes mode changes unreliable.
    messages = isolated_case_messages(grounded_prompt, message)
    yield detection_prefix + PROGRESS_MESSAGES.get(resolved_language, PROGRESS_MESSAGES["English"])
    try:
        response = requests.post(
            f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions",
            json={
                "model": "fieldmind-africa",
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 180,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
        answer_text = ""
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            token = event["choices"][0].get("delta", {}).get("content")
            if token:
                answer_text += token
                yield detection_prefix + sanitize_answer(answer_text, product_label, purchase_evidence, resolved_language)
        if not answer_text:
            yield "FieldMind could not produce a visible answer. Please retry once."
        elif sources:
            source_heading = SOURCES_HEADINGS.get(resolved_language, SOURCES_HEADINGS["English"])
            yield detection_prefix + sanitize_answer(answer_text, product_label, purchase_evidence, resolved_language).rstrip() + f"\n\n{source_heading}\n" + "\n".join(f"- {source}" for source in sources)
    except Exception as exc:
        yield f"The free CPU demo could not complete this request: {type(exc).__name__}. Please retry after a short wait."


CSS = """
.gradio-container {max-width: 980px !important; margin: auto !important;}
.hero {background: linear-gradient(135deg,#123c2d,#236d4b); color:white; padding:24px; border-radius:18px; margin-bottom:12px;}
.hero h1 {margin:0 0 8px 0; font-size:2rem;}
.badge {display:inline-block; background:#fff3bf; color:#5f4600; padding:5px 10px; border-radius:999px; font-weight:700;}
"""

with gr.Blocks(css=CSS, title="FieldMind Africa") as demo:
    gr.HTML(
        f"""<section class='hero'>
        <span class='badge'>FREE CLOUD DEMO · TRAINED FIELDMIND Q5 MODEL</span>
        <h1>🌱 FieldMind Africa</h1>
        <p><strong>Before you buy the chemical, ask FieldMind.</strong></p>
        <p>Free CPU cloud demo with verified English/Kiswahili purchase cards and local GGUF diagnosis mode. No paid API. Current model: {MODEL_LABEL}.</p>
        </section>"""
    )
    gr.Markdown(
        "Describe the crop, symptoms, field pattern, recent weather, crop stage, and the purchase you are considering. "
        "This is decision support—not a laboratory diagnosis, product label, or replacement for a local extension professional."
    )
    response_language = gr.Dropdown(
        choices=list(LANGUAGE_CHOICES),
        value="Automatic (detect my language)",
        label="Response language (verified: English and Kiswahili)",
    )
    decision_mode = gr.Dropdown(
        choices=list(DECISION_MODES),
        value="Chemical purchase plan",
        label="What decision do you need?",
    )
    country = gr.Dropdown(
        choices=list(COUNTRIES),
        value="Kenya",
        label="Country (controls the product registry)",
    )
    product_label = gr.Textbox(
        label="Exact product label text (optional; required for an exact dose)",
        placeholder="Paste the product name, active ingredient, formulation, crop/target, rate, PHI, REI and safety directions from the pack.",
        lines=3,
    )
    gr.ChatInterface(
        fn=answer,
        type="messages",
        chatbot=gr.Chatbot(height=500, type="messages", show_copy_button=True),
        textbox=gr.Textbox(
            placeholder="Example: My cassava leaves are curling after heavy rain. Should I buy fungicide?",
            lines=2,
            submit_btn="Ask FieldMind",
            stop_btn=True,
        ),
        additional_inputs=[response_language, decision_mode, country, product_label],
        examples=[
            ["My cassava leaves are curling with yellow-green mosaic patches after heavy rain. Should I buy fungicide?", "Automatic (detect my language)", "Chemical purchase plan", "Kenya", ""],
            ["Mahindi ya wiki tatu yana majani ya chini ya njano baada ya mvua nyingi. Ninunue CAN sasa?", "Automatic (detect my language)", "Fertilizer purchase plan", "Kenya", ""],
            ["Majani ya nyanya yana madoa ya kahawia yenye duara, yakianzia chini baada ya mvua. Ninunue dawa gani?", "Automatic (detect my language)", "Chemical purchase plan", "Kenya", ""],
        ],
        submit_btn="Ask FieldMind",
        save_history=False,
    )
    gr.Markdown(
        "**Verified scope:** automatic detection and complete replies are supported in English and Kiswahili. Matched demo cases are grounded in an IITA cassava IPM guide and university extension references for flooded maize and tomato leaf spots. "
        "The purchase flow can name registry-backed options and give a clear buy/shortlist/do-not-buy verdict. "
        "An exact dose is shown only from the exact registered product label supplied by the user, because rates differ by product and formulation."
    )
    gr.Markdown("Built for ADTC 2026 · CPU inference with llama.cpp · Apache-2.0 base model")

demo.queue(default_concurrency_limit=1, max_size=8).launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=False,
)
