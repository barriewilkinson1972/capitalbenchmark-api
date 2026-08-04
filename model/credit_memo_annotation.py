from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

ANNOTATION_SCHEMA = "credit_memo_annotation"
ANNOTATION_SCHEMA_VERSION = "1.3.1"
ANNOTATION_ENGINE_VERSION = "0.9.1"

SEVERITY_WEIGHTS = {
    "critical": 20,
    "high": 10,
    "medium": 5,
    "low": 2,
    "info": 0,
}


LLM_DIMENSIONS = ("reasoning", "fidelity", "tone")

CATEGORY_DIMENSION = {
    "factual_error": "fidelity",
    "unsupported_claim": "fidelity",
    "unsupported_claim_candidate": "fidelity",
    "fact_misstatement": "fidelity",
    "fabricated_fact": "fidelity",
    "fabricated_number": "fidelity",
    "certainty_inflation": "fidelity",
    "source_misattribution": "fidelity",
    "internal_contradiction": "fidelity",
    "policy_miss": "reasoning",
    "missing_information_miss": "reasoning",
    "rating_driver_confusion": "reasoning",
    "unsafe_recommendation": "reasoning",
    "omission": "reasoning",
    "correct_detection": "reasoning",
    "correct_inference": "reasoning",
    "overinclusive_policy_reference": "reasoning",
    "possible_false_positive": "reasoning",
    "imprecise_driver_language": "tone",
    "excessive_caveating": "tone",
    "excessive_confidence": "tone",
    "repetition": "tone",
    "boilerplate": "tone",
    "unprofessional_language": "tone",
    "generation_method_reference": "tone",
    "hidden_fact_assertion": "fidelity",
    "unavailable_input_miss": "reasoning",
}

DIMENSION_SEVERITY_WEIGHTS = {
    "reasoning": {"critical": 20, "high": 10, "medium": 5, "low": 2, "info": 0},
    "fidelity": {"critical": 25, "high": 12, "medium": 6, "low": 2, "info": 0},
    "tone": {"critical": 15, "high": 8, "medium": 4, "low": 2, "info": 0},
}

UNSUPPORTED_PHRASES = [
    "market leader",
    "leading provider",
    "established market presence",
    "strong market presence",
    "solid market presence",
    "strong operational footprint",
    "operational stability",
    "operational efficiency",
    "financial instability",
    "improving trend",
    "deterioration",
    "resilience",
    "resilient",
    "strategic importance",
    "urgent need",
    "best practices",
    "cash flow constraints",
    "operational losses",
    "approval is recommended",
]

POLICY_SYNONYMS = {
    "CP-01": ["CP-01", "minimum rating", "below the minimum rating"],
    "CP-02": ["CP-02", "maximum leverage", "4.0x"],
    "CP-03": ["CP-03", "severe leverage", "senior credit committee", "6.0x"],
    "CP-04": ["CP-04", "net leverage", "net debt/ebitda", "net debt / ebitda", "5.0x"],
    "CP-05": ["CP-05", "profitability", "negative profit margin", "negative profitability"],
    "CP-06": ["CP-06", "liquidity", "cash/debt", "cash / debt", "10%"],
    "CP-07": ["CP-07", "agency rating", "externally benchmarked", "external benchmarking"],
    "CP-08": ["CP-08", "data quality", "imputation", "incomplete"],
    "CP-09": ["CP-09", "existing exposure", "exposure is not supplied"],
    "CP-10": ["CP-10", "facility purpose", "general corporate purposes", "use of proceeds"],
}

RATING_TOKEN = r"(?:AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|D)"
RATING_RE = re.compile(rf"(?<![A-Z0-9])({RATING_TOKEN})(?![A-Z0-9])")

RATING_ASSERTION_PATTERNS = [
    re.compile(rf"(?:capital benchmark|cb|internal|indicative|model|proprietary)[^.!?;:]{{0,45}}?(?:rating|rating equivalent|assessment)[^.!?;:]{{0,20}}?(?<![A-Z0-9])(?P<rating>{RATING_TOKEN})(?![A-Z0-9+-])", re.I),
    re.compile(rf"(?:rating|rating equivalent|assessment)[^.!?;:]{{0,25}}?(?:is|of|:|=)?\s*(?<![A-Z0-9])(?P<rating>{RATING_TOKEN})(?![A-Z0-9+-])", re.I),
    re.compile(rf"(?:borrower|obligor|company|issuer|aal)[^.!?;:]{{0,30}}?(?:is|was|remains)?\s*(?:rated|assessed)[^.!?;:]{{0,12}}?(?<![A-Z0-9])(?P<rating>{RATING_TOKEN})(?![A-Z0-9+-])", re.I),
]

RATING_NON_ASSERTION_TERMS = [
    "minimum rating",
    "threshold",
    "below",
    "above",
    "or below",
    "or better",
    "policy",
    "trigger",
    "category",
    "single-b",
    "single b",
    "rating scale",
    "agency rating",
    "external rating",
    "benchmarking",
    "industry rating anchor",
    "country rating anchor",
    "industry credit-quality anchor",
    "country credit-quality anchor",
    "industry anchor",
    "country anchor",
]


# Equal-weight reference catalogue used to calculate information completeness.
# A case-specific catalogue may override this by supplying
# memo_payload["reference_item_catalogue"] or memo_context["reference_item_catalogue"].
DEFAULT_REFERENCE_ITEM_CATALOGUE: list[dict[str, Any]] = [
    {"item_id": "REF-001", "label": "Borrower name", "category": "borrower_identity", "visibility": "always", "source_paths": ["borrower.name", "borrower.company_name", "borrower.borrower_name"]},
    {"item_id": "REF-002", "label": "Ticker / obligor identifier", "category": "borrower_identity", "visibility": "always", "source_paths": ["borrower.ticker", "borrower.symbol", "borrower.obligor_id"]},
    {"item_id": "REF-003", "label": "Country", "category": "borrower_identity", "visibility": "always", "source_paths": ["borrower.country", "borrower.country_name", "borrower.domicile"]},
    {"item_id": "REF-004", "label": "Industry", "category": "borrower_identity", "visibility": "always", "source_paths": ["borrower.industry", "borrower.industry_name", "borrower.sector"]},
    {"item_id": "REF-005", "label": "Business description", "category": "borrower_identity", "visibility": "always", "source_paths": ["borrower.business_summary", "borrower.business_description", "borrower.description"]},
    {"item_id": "REF-006", "label": "Facility type", "category": "credit_request", "visibility": "always", "source_paths": ["credit_request.facility_type", "credit_request.request_type"]},
    {"item_id": "REF-007", "label": "Facility amount", "category": "credit_request", "visibility": "always", "source_paths": ["credit_request.proposed_exposure_usd", "credit_request.facility_amount_usd", "credit_request.amount_usd", "credit_request.amount"]},
    {"item_id": "REF-008", "label": "Facility purpose / use of proceeds", "category": "credit_request", "visibility": "always", "source_paths": ["credit_request.purpose", "credit_request.use_of_proceeds"]},
    {"item_id": "REF-009", "label": "Existing exposure", "category": "credit_request", "visibility": "always", "source_paths": ["credit_request.existing_exposure_usd", "credit_request.existing_exposure"]},
    {"item_id": "REF-010", "label": "Facility tenor", "category": "credit_request", "visibility": "always", "source_paths": ["credit_request.tenor_years", "credit_request.tenor"]},
    {"item_id": "REF-011", "label": "Security status / package", "category": "credit_request", "visibility": "always", "source_paths": ["credit_request.secured", "credit_request.security_package", "credit_request.security"]},
    {"item_id": "REF-012", "label": "Seniority", "category": "credit_request", "visibility": "always", "source_paths": ["credit_request.seniority"]},
    {"item_id": "REF-013", "label": "Revenue", "category": "financial", "visibility": "financials", "source_keys": ["revenue", "total_revenue"]},
    {"item_id": "REF-014", "label": "EBITDA", "category": "financial", "visibility": "financials", "source_keys": ["ebitda", "adjusted_ebitda"]},
    {"item_id": "REF-015", "label": "Total debt", "category": "financial", "visibility": "financials", "source_keys": ["total_debt", "debt"]},
    {"item_id": "REF-016", "label": "Net debt", "category": "financial", "visibility": "financials", "source_keys": ["net_debt"]},
    {"item_id": "REF-017", "label": "Cash and cash equivalents", "category": "financial", "visibility": "financials", "source_keys": ["cash", "cash_and_cash_equivalents", "cash_equivalents"]},
    {"item_id": "REF-018", "label": "Debt / EBITDA", "category": "financial", "visibility": "financials", "source_keys": ["debt_ebitda", "debt_to_ebitda"]},
    {"item_id": "REF-019", "label": "Net Debt / EBITDA", "category": "financial", "visibility": "financials", "source_keys": ["net_debt_ebitda", "net_debt_to_ebitda"]},
    {"item_id": "REF-020", "label": "Capital Benchmark internal rating", "category": "rating", "visibility": "rating", "source_paths": ["capital_benchmark_rating.cb_rating"]},
    {"item_id": "REF-021", "label": "Probability of default", "category": "rating", "visibility": "rating", "source_paths": ["capital_benchmark_rating.cb_pd"]},
    {"item_id": "REF-022", "label": "Expected loss", "category": "exposure_analytics", "visibility": "expected_loss", "source_paths": ["exposure_analytics.base_expected_loss_on_proposed_exposure", "exposure_analytics.expected_loss"]},
    {"item_id": "POL-001", "label": "CP-01 Minimum rating rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-01"},
    {"item_id": "POL-002", "label": "CP-02 Maximum leverage rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-02"},
    {"item_id": "POL-003", "label": "CP-03 Severe leverage rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-03"},
    {"item_id": "POL-004", "label": "CP-04 Net leverage review rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-04"},
    {"item_id": "POL-005", "label": "CP-05 Profitability rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-05"},
    {"item_id": "POL-006", "label": "CP-06 Liquidity rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-06"},
    {"item_id": "POL-007", "label": "CP-07 External rating benchmark rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-07"},
    {"item_id": "POL-008", "label": "CP-08 Data quality rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-08"},
    {"item_id": "POL-009", "label": "CP-09 Existing exposure rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-09"},
    {"item_id": "POL-010", "label": "CP-10 Facility purpose rule", "category": "policy_rule", "visibility": "policy", "supply_basis": "visibility_only", "policy_id": "CP-10"},
]



def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if value == "" or value.lower() in {"nan", "none", "null", "n/a"}:
                return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _format_percent(value: Any) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number * 100:.2f}%"


def _format_currency(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000_000:
        return f"{sign}${number / 1_000_000_000_000:.2f}tn"
    if number >= 1_000_000_000:
        return f"{sign}${number / 1_000_000_000:.1f}bn"
    if number >= 1_000_000:
        return f"{sign}${number / 1_000_000:.1f}mn"
    return f"{sign}${number:,.0f}"


def _flatten(value: Any) -> str:
    parts: list[str] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            parts.append(str(x))

    walk(value)
    return "\n".join(parts)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_any(text_norm: str, terms: list[str]) -> bool:
    return any(_normalise(term) in text_norm for term in terms if term)


def _snippet(text: str, needle: str, chars: int = 110) -> str:
    idx = _normalise(text).find(_normalise(needle))
    if idx < 0:
        return needle
    start = max(0, idx - chars // 2)
    end = min(len(text), idx + len(needle) + chars // 2)
    return text[start:end].replace("\n", " ").strip()


def _annotation(
    annotations: list[dict[str, Any]],
    *,
    category: str,
    severity: str,
    title: str,
    detail: str,
    expected: Any = None,
    observed: Any = None,
    evidence: Any = None,
    policy_id: str | None = None,
    location: str | None = None,
    scorer: str = "deterministic",
    dimension: str | None = None,
    source_type: str = "llm",
    score_impact: float | None = None,
) -> None:
    annotations.append(
        {
            "annotation_id": f"A{len(annotations) + 1:04d}",
            "category": category,
            "dimension": dimension or CATEGORY_DIMENSION.get(category),
            "severity": severity,
            "title": title,
            "detail": detail,
            "expected": expected,
            "observed": observed,
            "evidence": evidence,
            "policy_id": policy_id,
            "location": location,
            "source_type": source_type,
            "score_impact": score_impact,
            "scorer": scorer,
            "status": "draft",
            "public_label": _public_label(category, severity),
        }
    )


def _public_label(category: str, severity: str) -> str:
    if severity in {"critical", "high"}:
        return "Material Error ❌"
    if category in {"unsupported_claim", "omission", "policy_miss", "rating_driver_confusion"}:
        return "Important Limitation ⚠"
    if category in {"strength", "correct_detection"}:
        return "Strength ✓"
    return "Review Note"


def _path_value(root: Any, dotted_path: str) -> Any:
    value = root
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _is_supplied_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    if isinstance(value, str):
        return bool(_clean_text(value))
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _find_key_recursive(value: Any, candidate_keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            key_norm = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if key_norm in candidate_keys and _is_supplied_value(child):
                return child
        for child in value.values():
            found = _find_key_recursive(child, candidate_keys)
            if _is_supplied_value(found):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key_recursive(child, candidate_keys)
            if _is_supplied_value(found):
                return found
    return None


def _reference_item_visible(item: dict[str, Any], profile: dict[str, Any]) -> bool:
    visibility = _clean_text(item.get("visibility"), "always").lower()
    if visibility == "always":
        return True
    if visibility == "financials":
        return bool(profile.get("financials_visible"))
    if visibility == "rating":
        return bool(profile.get("rating_visible"))
    if visibility == "expected_loss":
        return bool(profile.get("expected_loss_expected"))
    if visibility == "policy":
        return bool(profile.get("policy_visible"))
    return False


def _reference_item_value(item: dict[str, Any], memo_context: dict[str, Any]) -> Any:
    # Some architectural inputs are supplied as a complete package. Policy is
    # the first example: when policy visibility is enabled, every rule in the
    # standard policy pack is supplied, whether or not that rule is triggered
    # for the current obligor. These items therefore do not depend on a
    # borrower-specific value appearing in memo_context.
    if _clean_text(item.get("supply_basis")).lower() == "visibility_only":
        return True

    for path in item.get("source_paths") or []:
        value = _path_value(memo_context, str(path))
        if _is_supplied_value(value):
            return value

    source_keys = {
        re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
        for key in (item.get("source_keys") or [])
        if key
    }
    if source_keys:
        return _find_key_recursive(memo_context, source_keys)

    return None


def _reference_item_catalogue(memo_payload: dict[str, Any], memo_context: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = memo_payload.get("reference_item_catalogue") or memo_context.get("reference_item_catalogue")
    if isinstance(candidate, dict):
        candidate = candidate.get("items")
    if isinstance(candidate, list):
        cleaned = [dict(item) for item in candidate if isinstance(item, dict)]
        if cleaned:
            return cleaned
    return [dict(item) for item in DEFAULT_REFERENCE_ITEM_CATALOGUE]


def _build_reference_coverage(memo_payload: dict[str, Any], memo_context: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    catalogue = _reference_item_catalogue(memo_payload, memo_context)
    supplied_items: list[dict[str, Any]] = []
    missing_items: list[dict[str, Any]] = []
    for item in catalogue:
        public_item = {"item_id": item.get("item_id"), "label": item.get("label"), "category": item.get("category")}
        visible = _reference_item_visible(item, profile)
        value = _reference_item_value(item, memo_context) if visible else None
        if visible and _is_supplied_value(value):
            supplied_items.append(public_item)
        else:
            missing_items.append({**public_item, "missing_reason": "not_exposed_by_architecture" if not visible else "authoritative_value_not_available"})
    total_count = len(catalogue)
    supplied_count = len(supplied_items)
    score = round(100 * supplied_count / total_count, 1) if total_count else 100.0
    return {
        "catalogue_version": "1.1.0",
        "scoring_method": "equal_weight_count",
        "score": score,
        "total_reference_items": total_count,
        "supplied_reference_items": supplied_count,
        "missing_reference_items": len(missing_items),
        "supplied_items": supplied_items,
        "missing_items": missing_items,
    }


def _reference_coverage_annotation(coverage: dict[str, Any]) -> dict[str, Any]:
    supplied_count = coverage.get("supplied_reference_items", 0)
    total_count = coverage.get("total_reference_items", 0)
    missing_count = coverage.get("missing_reference_items", 0)
    return {
        "annotation_id": "COV-001",
        "annotation_type": "reference_coverage",
        "category": "reference_coverage",
        "severity": "info",
        "title": "Reference Coverage",
        "detail": f"{supplied_count} of {total_count} reference items supplied. Supplied {supplied_count}; Missing {missing_count}.",
        "score_dimension": "information_completeness",
        "summary": {"score": coverage.get("score"), "total": total_count, "supplied": supplied_count, "missing": missing_count},
        "supplied_items": coverage.get("supplied_items", []),
        "missing_items": coverage.get("missing_items", []),
        "location": None,
        "scorer": "deterministic_reference_coverage",
        "status": "draft",
        "public_label": "Reference Coverage",
    }



def _build_source_manifest(memo_payload: dict[str, Any]) -> dict[str, Any]:
    experiment_config = memo_payload.get("experiment_config") or {}
    deterministic_sections = set(experiment_config.get("deterministic_sections") or [])
    section_sources = {
        "request_summary": "llm", "executive_summary": "llm",
        "business_profile": "llm", "rating_assessment": "llm",
        "financial_risk_assessment": "llm", "rating_driver_commentary": "llm",
        "peer_context": "llm", "policy_compliance_assessment": "llm",
        "policy_breaches": "llm", "policy_required_actions": "llm",
        "policy_missing_information": "llm", "policy_escalation_assessment": "llm",
        "strengths": "llm", "key_risks": "llm",
        "questions_for_relationship_manager": "llm",
        "conclusion_recommendation": "llm",
        "data_quality_and_limitations": "llm",
        "credit_request_table": "deterministic", "rating_table": "deterministic",
        "financial_table": "deterministic", "exposure_analytics_table": "deterministic",
    }
    if "policy_assessment" in deterministic_sections:
        for key in ["policy_compliance_assessment", "policy_breaches",
                    "policy_required_actions", "policy_missing_information",
                    "policy_escalation_assessment"]:
            section_sources[key] = "deterministic"
    return {
        "manifest_version": "1.0.0",
        "attribution_level": "section",
        "section_sources": section_sources,
        "deterministic_sections": sorted(deterministic_sections),
        "frontend_hint": {
            "toggle_label": "Source",
            "llm_label": "LLM generated",
            "deterministic_label": "Deterministically generated",
        },
    }


def _llm_narrative_for_scoring(narrative: dict[str, Any], source_manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    section_sources = source_manifest.get("section_sources") or {}
    llm_sections = {k: v for k, v in narrative.items() if section_sources.get(k, "llm") == "llm"}
    return llm_sections, _flatten(llm_sections)


def _check_tone(annotations: list[dict[str, Any]], narrative: dict[str, Any], llm_text: str) -> dict[str, Any]:
    text_norm = _normalise(llm_text)
    sentences = _sentence_windows(llm_text)
    unavailable_terms = ["cannot be assessed", "cannot be determined", "not provided",
                         "not available", "insufficient information", "unable to assess"]
    caveat_hits = sum(text_norm.count(term) for term in unavailable_terms)
    if caveat_hits >= 7:
        _annotation(annotations, category="excessive_caveating", severity="medium",
                    title="Excessive repetition of information limitations",
                    detail="The memo repeatedly restates unavailable information rather than consolidating limitations into a concise, decision-useful explanation.",
                    observed={"caveat_phrase_count": caveat_hits}, location="llm_narrative")
    elif caveat_hits >= 4:
        _annotation(annotations, category="excessive_caveating", severity="low",
                    title="Repeated caveating reduces concision",
                    detail="Several sections repeat substantially similar information limitations.",
                    observed={"caveat_phrase_count": caveat_hits}, location="llm_narrative")
    confidence_terms = ["clearly demonstrates", "undoubtedly", "certainly", "will default",
                        "no material risk", "fully compliant", "approval is recommended",
                        "should be approved"]
    confidence_hits = [term for term in confidence_terms if term in text_norm]
    if confidence_hits:
        _annotation(annotations, category="excessive_confidence", severity="medium",
                    title="Confidence exceeds the evidential basis",
                    detail="The prose uses definitive decision or risk language that is stronger than appropriate for the supplied evidence.",
                    observed=confidence_hits, location="llm_narrative")
    generation_terms = [
        "language model",
        "large language model",
        "system prompt",
        "user prompt",
        "the prompt",
        "prompt instructions",
        "hidden context",
        "benchmark experiment",
        "generation method",
    ]
    generation_hits = [term for term in generation_terms if term in text_norm]
    if generation_hits:
        _annotation(annotations, category="generation_method_reference", severity="medium",
                    title="Memo refers to generation mechanics",
                    detail="Credit-memo prose should not mention prompts, language models, hidden context or benchmark mechanics.",
                    observed=generation_hits, location="llm_narrative")
    long_sentences = [s for s in sentences if len(s.split()) > 45]
    if len(long_sentences) >= 3:
        _annotation(annotations, category="repetition", severity="low",
                    title="Prose is unnecessarily dense",
                    detail="Several sentences are unusually long for committee-facing credit prose.",
                    observed={"long_sentence_count": len(long_sentences)},
                    evidence={"examples": long_sentences[:3]}, location="llm_narrative")
    return {"caveat_phrase_count": caveat_hits,
            "confidence_terms_found": confidence_hits,
            "generation_method_terms_found": generation_hits,
            "long_sentence_count": len(long_sentences)}


def _positive_reasoning_findings(annotations: list[dict[str, Any]], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    policy = diagnostics.get("policy_detection") or {}
    if policy.get("policy_detection_scored"):
        expected = set(policy.get("expected_policy_ids") or [])
        observed = set(policy.get("observed_triggered_policy_ids") or [])
        for policy_id in sorted(expected & observed):
            findings.append({"finding_id": f"RPOS-{len(findings)+1:03d}",
                             "dimension": "reasoning", "category": "correct_detection",
                             "policy_id": policy_id,
                             "title": f"Correctly identified triggered policy rule {policy_id}"})
    missing_info = diagnostics.get("missing_information") or {}
    for item in missing_info.get("found_missing_information") or []:
        findings.append({"finding_id": f"RPOS-{len(findings)+1:03d}",
                         "dimension": "reasoning", "category": "correct_detection",
                         "title": f"Correctly identified missing information: {item}"})
    return findings


def _score_dimension(annotations: list[dict[str, Any]], dimension: str) -> tuple[float, list[dict[str, Any]]]:
    relevant = [a for a in annotations if a.get("dimension") == dimension
                and a.get("severity") != "info"
                and a.get("source_type", "llm") == "llm"]
    weights = DIMENSION_SEVERITY_WEIGHTS[dimension]
    penalty = sum(weights.get(a.get("severity", "low"), 2) for a in relevant)
    return float(max(0, 100 - penalty)), relevant


def _evaluation_profile(experiment_config: dict[str, Any]) -> dict[str, Any]:
    context_mode = _clean_text(experiment_config.get("context_mode")).lower()
    policy_mode = _clean_text(experiment_config.get("policy_mode")).lower()

    rating_expected = context_mode in {
        "rating_only",
        "rating_and_financials",
        "full",
    }
    financials_visible = context_mode in {
        "financials_only",
        "rating_and_financials",
        "full",
    }

    # Policy visibility and LLM policy-reasoning attribution are deliberately
    # separate. Both policy modes receive completeness credit for the supplied
    # policy pack, but only llm_evaluated asks the LLM to derive the findings.
    policy_visible = policy_mode in {
        "llm_evaluated",
        "deterministic_evaluated",
    }
    policy_trigger_detection_expected = policy_mode == "llm_evaluated"

    return {
        "context_mode": context_mode or None,
        "policy_mode": policy_mode or None,
        "rating_visible": rating_expected,
        "rating_expected": rating_expected,
        "pd_expected": rating_expected,
        "financials_visible": financials_visible,
        "expected_loss_expected": context_mode == "full",
        "policy_visible": policy_visible,
        "policy_trigger_detection_expected": (
            policy_trigger_detection_expected
        ),
    }


def _sentence_windows(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]


def _extract_borrower_rating_assertions(text: str) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _sentence_windows(text):
        sentence_norm = _normalise(sentence)
        for pattern in RATING_ASSERTION_PATTERNS:
            for match in pattern.finditer(sentence):
                raw_rating = match.group("rating")
                rating = raw_rating.upper()

                # The patterns are case-insensitive to accept forms such as
                # "bb-", but a one-letter grade must be written as an explicit
                # uppercase rating token. This prevents ordinary articles such
                # as "a preliminary assessment" from being classified as A.
                if len(raw_rating) == 1 and raw_rating != raw_rating.upper():
                    continue

                start, end = match.span("rating")
                local = _normalise(
                    sentence[
                        max(0, start - 55):
                        min(len(sentence), end + 55)
                    ]
                )
                if any(term in local for term in RATING_NON_ASSERTION_TERMS):
                    # Allow a clear CB/internal rating assertion even when the same sentence
                    # later compares it with a policy threshold.
                    prefix = _normalise(sentence[max(0, start - 50):start])
                    if not any(term in prefix for term in ["cb rating", "capital benchmark", "internal rating", "indicative rating", "model indicates"]):
                        continue
                key = (rating, sentence)
                if key in seen:
                    continue
                seen.add(key)
                assertions.append({"rating": rating, "sentence": sentence, "char_start": start, "char_end": end})
    return assertions


def _classify_policy_reference(
    sentence: str,
    policy_id: str,
) -> str:
    norm = _normalise(sentence)

    not_assessable_patterns = [
        "not assessable",
        "cannot be assessed",
        "unable to assess",
        "not provided",
        "not available",
        "insufficient information",
        "cannot be determined",
        "information is missing",
        "details are missing",
        "details are required",
        "needs to be provided",
        "must be provided",
    ]
    negative_patterns = [
        "not triggered",
        "not breached",
        "no breach",
        "satisfied",
        "compliant",
        "no issue",
        "does not trigger",
        "within policy",
        "meets policy",
    ]
    conditional_patterns = [
        "if ",
        "would require",
        "would trigger",
        "subject to",
        "once provided",
        "upon receipt",
        "if evidenced",
        "to assess",
        "to test",
    ]
    triggered_patterns = [
        "breach",
        "breached",
        "triggered",
        "not met",
        "below the minimum",
        "below the required minimum",
        "does not comply",
        "non-compliant",
        "noncompliant",
        "exceeds",
        "requires exception",
        "exception approval required",
        "exception approval is mandatory",
        "exception is mandatory",
        "must be obtained",
        "is missing",
        "requires enhanced review",
        "senior credit committee is required",
        "senior credit committee approval is required",
        "escalation to senior credit committee is required",
    ]

    # For missing-information policies, a clear statement that the required
    # information is absent should count as an affirmative trigger rather than
    # merely "not assessable".
    missing_information_policy = policy_id in {"CP-09", "CP-10"}
    if missing_information_policy and any(
        phrase in norm
        for phrase in [
            "not provided",
            "not available",
            "information is missing",
            "details are missing",
            "details are required",
            "needs to be provided",
            "must be provided",
            "general corporate purposes",
        ]
    ):
        return "triggered"

    if any(x in norm for x in negative_patterns):
        return "not_triggered"
    if any(x in norm for x in triggered_patterns):
        return "triggered"
    if any(x in norm for x in not_assessable_patterns):
        return "not_assessable"
    if any(x in norm for x in conditional_patterns):
        return "conditional"
    return "mentioned"

def _policy_references(text: str) -> list[dict[str, Any]]:
    """Extract policy references from explicit IDs and natural-language terms."""
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for sentence in _sentence_windows(text):
        sentence_upper = sentence.upper()
        sentence_norm = _normalise(sentence)

        matched_ids = set(
            re.findall(r"\bCP-\d{2}\b", sentence_upper)
        )

        # Credit memos do not need to reproduce machine-readable policy IDs.
        # Treat a rule as referenced where the sentence uses one of its
        # configured ordinary-language synonyms.
        for policy_id, terms in POLICY_SYNONYMS.items():
            if _contains_any(sentence_norm, terms):
                matched_ids.add(policy_id)

        for policy_id in sorted(matched_ids):
            status = _classify_policy_reference(sentence, policy_id)
            key = (policy_id, status, sentence)
            if key in seen:
                continue
            seen.add(key)

            matched_terms = [
                term
                for term in POLICY_SYNONYMS.get(policy_id, [])
                if _normalise(term) in sentence_norm
            ]

            refs.append({
                "policy_id": policy_id,
                "status": status,
                "sentence": sentence,
                "matched_by": (
                    "explicit_policy_id"
                    if policy_id in set(
                        re.findall(r"\bCP-\d{2}\b", sentence_upper)
                    )
                    else "natural_language"
                ),
                "matched_terms": matched_terms,
            })

    return refs

def _expected_policy_ids(memo_context: dict[str, Any]) -> list[str]:
    policy_eval = memo_context.get("policy_evaluation") or {}
    ids = policy_eval.get("triggered_policy_ids") or []
    return [str(x) for x in ids if x]


def _visible_policy_ids(text_norm: str) -> set[str]:
    found = set(re.findall(r"\bCP-\d{2}\b", text_norm.upper()))
    for policy_id, terms in POLICY_SYNONYMS.items():
        if _contains_any(text_norm, terms):
            found.add(policy_id)
    return found


def _narrative_policy_text(narrative: dict[str, Any]) -> str:
    keys = [
        "policy_compliance_assessment",
        "policy_breaches",
        "policy_required_actions",
        "policy_missing_information",
        "policy_escalation_assessment",
        "conclusion_recommendation",
        "executive_summary",
    ]
    return "\n".join(_flatten(narrative.get(k)) for k in keys if k in narrative)


def _check_core_rating_and_pd(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative_text: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    rating = memo_context.get("capital_benchmark_rating", {})
    expected_rating = _clean_text(
        rating.get("cb_rating")
    ).upper()
    expected_pd = _safe_float(rating.get("cb_pd"))

    assertions = _extract_borrower_rating_assertions(
        narrative_text
    )
    asserted_ratings = sorted(
        {item["rating"] for item in assertions}
    )

    rating_visible = bool(profile.get("rating_expected"))
    pd_visible = bool(profile.get("pd_expected"))

    if assertions and not rating_visible:
        _annotation(
            annotations,
            category="hidden_fact_assertion",
            severity="high",
            title="Borrower rating asserted despite rating being hidden",
            detail=(
                "The architecture did not expose the borrower rating. Any "
                "borrower-rating assertion is therefore unsupported, even if "
                "it happens to match the hidden benchmark value."
            ),
            expected="No borrower-rating assertion",
            observed=asserted_ratings,
            evidence={"borrower_rating_assertions": assertions},
            location="narrative",
        )
    elif expected_rating:
        wrong_ratings = sorted(
            rating_value
            for rating_value in asserted_ratings
            if rating_value != expected_rating
        )
        if wrong_ratings:
            _annotation(
                annotations,
                category="factual_error",
                severity="high",
                title=(
                    "Conflicting borrower credit rating asserted in "
                    "LLM prose"
                ),
                detail=(
                    "The prose explicitly asserts a borrower/internal rating "
                    "that does not match the deterministic CB rating. Policy "
                    "thresholds and rating-anchor references are excluded."
                ),
                expected=expected_rating,
                observed=wrong_ratings,
                evidence={
                    "borrower_rating_assertions": assertions
                },
                location="narrative",
            )
        elif rating_visible and expected_rating not in asserted_ratings:
            _annotation(
                annotations,
                category="omission",
                severity="medium",
                title="CB rating not explicitly mentioned",
                detail=(
                    "The CB rating was visible in this experimental context "
                    "and should normally appear in the executive summary or "
                    "rating assessment."
                ),
                expected=expected_rating,
                observed="no borrower-rating assertion found",
                evidence={
                    "borrower_rating_assertions": assertions
                },
                location="narrative",
            )

    pd_assertion_pattern = re.compile(
        r"\b(?:pd|probability of default)\b[^.!?;:]{0,35}"
        r"(?P<value>\d+(?:\.\d+)?\s*%)",
        re.I,
    )
    pd_assertions = [
        {
            "value": match.group("value"),
            "sentence": sentence,
        }
        for sentence in _sentence_windows(narrative_text)
        for match in pd_assertion_pattern.finditer(sentence)
    ]

    if pd_assertions and not pd_visible:
        _annotation(
            annotations,
            category="hidden_fact_assertion",
            severity="high",
            title="Probability of default asserted despite PD being hidden",
            detail=(
                "The architecture did not expose the PD. Any numerical PD "
                "assertion is therefore unsupported."
            ),
            expected="No PD assertion",
            observed=pd_assertions,
            location="narrative",
        )
    elif expected_pd is not None and pd_visible:
        expected_pd_text = _format_percent(expected_pd)
        pd_variants = {
            expected_pd_text,
            expected_pd_text.replace(".00%", "%"),
            f"{expected_pd * 100:.1f}%",
        }
        if not any(
            variant in narrative_text
            for variant in pd_variants
        ):
            _annotation(
                annotations,
                category="omission",
                severity="low",
                title="CB PD not explicitly mentioned",
                detail=(
                    "The CB PD was visible in this experimental context and "
                    "should normally appear in the executive summary or "
                    "rating assessment."
                ),
                expected=expected_pd_text,
                observed="not found in narrative prose",
                location="narrative",
            )

    return {
        "expected_rating": expected_rating or None,
        "borrower_rating_assertions": assertions,
        "asserted_borrower_ratings": asserted_ratings,
        "pd_assertions": pd_assertions,
        "rating_scored": rating_visible,
        "pd_scored": pd_visible,
    }

def _check_expected_loss(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative_text: str,
    profile: dict[str, Any],
) -> None:
    if not profile.get("expected_loss_expected"):
        return
    exposure = memo_context.get("exposure_analytics", {})
    expected_el = _safe_float(exposure.get("base_expected_loss_on_proposed_exposure"))
    if expected_el is None:
        return
    expected_text = _format_currency(expected_el)
    if expected_text not in narrative_text and str(int(round(expected_el))) not in narrative_text.replace(",", ""):
        _annotation(
            annotations,
            category="omission",
            severity="low",
            title="Expected loss not mentioned or not easy to verify",
            detail="Expected loss is deterministic and should be stated where exposure analytics are discussed.",
            expected=expected_text,
            observed="not found in narrative prose",
            location="executive_summary_or_request_summary",
        )


def _check_policy_detection(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = set(_expected_policy_ids(memo_context))
    policy_text = _narrative_policy_text(narrative)
    refs = _policy_references(policy_text)

    statuses_by_id: dict[str, set[str]] = {}
    evidence_by_id: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        policy_id = ref["policy_id"]
        statuses_by_id.setdefault(policy_id, set()).add(ref["status"])
        evidence_by_id.setdefault(policy_id, []).append(ref)

    observed_triggered = {
        policy_id for policy_id, statuses in statuses_by_id.items()
        if "triggered" in statuses
    }
    observed_considered = set(statuses_by_id)
    should_score = bool(profile.get("policy_trigger_detection_expected"))

    if should_score:
        for policy_id in sorted(expected_ids):
            if policy_id not in observed_triggered:
                observed_statuses = sorted(statuses_by_id.get(policy_id, set())) or ["not mentioned"]
                _annotation(
                    annotations,
                    category="policy_miss",
                    severity="high" if policy_id == "CP-03" else "medium",
                    title=f"Expected triggered policy finding not correctly stated: {policy_id}",
                    detail="The deterministic policy evaluation triggered this rule, but the memo did not state it as an affirmative triggered finding. Mere mentions, conditional statements and 'not assessable' references do not receive trigger credit.",
                    expected={"policy_id": policy_id, "status": "triggered"},
                    observed={"statuses": observed_statuses},
                    evidence={"memo_references": evidence_by_id.get(policy_id, []), "ground_truth": memo_context.get("policy_evaluation", {}).get("triggered_policies", [])},
                    policy_id=policy_id,
                    location="policy_prose",
                )

        for policy_id in sorted(observed_triggered - expected_ids):
            if policy_id == "CP-02" and "CP-03" in expected_ids:
                category = "overinclusive_policy_reference"
                severity = "info"
                detail = (
                    "The memo affirmatively cited CP-02 alongside the more "
                    "severe CP-03 trigger. This is treated as over-inclusive "
                    "rather than a material error."
                )
            else:
                category = "possible_false_positive"
                severity = "medium"
                detail = (
                    "The memo affirmatively states this policy as triggered, "
                    "but it is absent from the deterministic triggered-policy "
                    "list. This is treated as an incorrect policy conclusion."
                )
            _annotation(
                annotations,
                category=category,
                severity=severity,
                title=f"Additional triggered policy reference: {policy_id}",
                detail=detail,
                expected=sorted(expected_ids),
                observed=evidence_by_id.get(policy_id, []),
                policy_id=policy_id,
                location="policy_prose",
            )

        policy_eval = memo_context.get("policy_evaluation") or {}
        if policy_eval.get("requires_senior_credit_committee"):
            policy_text_norm = _normalise(policy_text)
            if not _contains_any(policy_text_norm, ["senior credit committee", "senior committee exception", "senior credit exception"]):
                _annotation(
                    annotations,
                    category="policy_miss",
                    severity="high",
                    title="Senior credit committee escalation not clearly stated",
                    detail="The deterministic policy evaluation requires senior credit committee exception approval.",
                    expected="senior credit committee exception approval",
                    observed="not clearly found",
                    policy_id="CP-03",
                    location="policy_escalation_assessment",
                )

    return {
        "policy_detection_scored": should_score,
        "expected_policy_ids": sorted(expected_ids) if should_score else [],
        "ground_truth_policy_ids_unscored": sorted(expected_ids) if not should_score else [],
        "observed_policy_ids": sorted(observed_considered),
        "observed_triggered_policy_ids": sorted(observed_triggered),
        "policy_reference_statuses": {
            k: sorted(v) for k, v in sorted(statuses_by_id.items())
        },
        "policy_reference_evidence": refs,
        "missed_policy_ids": sorted(expected_ids - observed_triggered) if should_score else [],
        "additional_policy_ids": sorted(observed_triggered - expected_ids) if should_score else [],
    }

def _check_missing_information(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative: dict[str, Any],
) -> dict[str, Any]:
    request = memo_context.get("credit_request", {})
    expected_missing = []

    if _safe_float(request.get("existing_exposure_usd")) is None:
        expected_missing.append("existing exposure")
    if _clean_text(request.get("purpose")).lower() in {"", "general_corporate_purposes", "general corporate purposes", "gcp", "other"}:
        expected_missing.append("specific facility purpose / use of proceeds")
    if _safe_float(request.get("tenor_years")) is None:
        expected_missing.append("tenor")
    if request.get("secured") is None:
        expected_missing.append("security package / secured status")
    if not _clean_text(request.get("seniority")):
        expected_missing.append("seniority")

    text = _normalise(_flatten({
        "policy_missing_information": narrative.get("policy_missing_information"),
        "questions_for_relationship_manager": narrative.get("questions_for_relationship_manager"),
        "data_quality_and_limitations": narrative.get("data_quality_and_limitations"),
        "request_summary": narrative.get("request_summary"),
        "conclusion_recommendation": narrative.get("conclusion_recommendation"),
    }))

    found = []
    missing = []
    checks = {
        "existing exposure": ["existing exposure"],
        "specific facility purpose / use of proceeds": ["use of proceeds", "facility purpose", "general corporate purposes", "purpose"],
        "tenor": ["tenor"],
        "security package / secured status": ["security", "secured", "security package"],
        "seniority": ["seniority", "senior unsecured", "senior secured"],
    }
    for item in expected_missing:
        if _contains_any(text, checks[item]):
            found.append(item)
        else:
            missing.append(item)
            _annotation(
                annotations,
                category="missing_information_miss",
                severity="medium" if item == "existing exposure" else "low",
                title=f"Missing information not highlighted: {item}",
                detail="The request context lacks this information, but the LLM did not clearly flag it as a follow-up item.",
                expected=item,
                observed="not found in missing-information or RM-question text",
                location="policy_missing_information/questions_for_relationship_manager",
            )

    return {
        "expected_missing_information": expected_missing,
        "found_missing_information": found,
        "missed_missing_information": missing,
    }


def _input_status_map(
    memo_payload: dict[str, Any],
) -> dict[str, str]:
    experiment_config = memo_payload.get(
        "experiment_config"
    ) or {}
    coverage = (
        experiment_config.get("input_data_coverage")
        or memo_payload.get("input_data_coverage")
        or {}
    )
    return {
        str(item.get("item_id")): str(
            item.get("status", "")
        ).lower()
        for item in (coverage.get("item_statuses") or [])
        if item.get("item_id")
    }


def _check_unavailable_inputs(
    annotations: list[dict[str, Any]],
    memo_payload: dict[str, Any],
    narrative: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    status_map = _input_status_map(memo_payload)
    text = _normalise(_flatten(narrative))

    checks = {
        "REF-013": {
            "label": "revenue",
            "visible": profile.get("financials_visible"),
            "terms": ["revenue", "sales"],
        },
        "REF-014": {
            "label": "EBITDA",
            "visible": profile.get("financials_visible"),
            "terms": ["ebitda"],
        },
        "REF-015": {
            "label": "total debt",
            "visible": profile.get("financials_visible"),
            "terms": ["total debt", "debt amount"],
        },
        "REF-016": {
            "label": "net debt",
            "visible": profile.get("financials_visible"),
            "terms": ["net debt"],
        },
        "REF-017": {
            "label": "cash",
            "visible": profile.get("financials_visible"),
            "terms": ["cash balance", "cash and cash equivalents", "cash"],
        },
        "REF-018": {
            "label": "Debt / EBITDA",
            "visible": profile.get("financials_visible"),
            "terms": ["debt / ebitda", "debt/ebitda"],
        },
        "REF-019": {
            "label": "Net Debt / EBITDA",
            "visible": profile.get("financials_visible"),
            "terms": ["net debt / ebitda", "net debt/ebitda"],
        },
        "REF-020": {
            "label": "Capital Benchmark rating",
            "visible": profile.get("rating_visible"),
            "terms": ["rating unavailable", "rating not available", "rating not provided"],
        },
        "REF-021": {
            "label": "probability of default",
            "visible": profile.get("pd_expected"),
            "terms": ["pd unavailable", "probability of default unavailable", "pd not available"],
        },
        "REF-022": {
            "label": "expected loss",
            "visible": profile.get("expected_loss_expected"),
            "terms": ["expected loss unavailable", "expected loss not available"],
        },
    }

    expected_unavailable = []
    acknowledged = []
    missed = []

    for item_id, spec in checks.items():
        if not spec["visible"]:
            continue
        if status_map.get(item_id) != "unavailable":
            continue

        expected_unavailable.append(item_id)
        if _contains_any(text, spec["terms"]):
            acknowledged.append(item_id)
            continue

        missed.append(item_id)
        _annotation(
            annotations,
            category="unavailable_input_miss",
            severity="medium"
            if item_id in {"REF-020", "REF-021", "REF-022"}
            else "low",
            title=f"Unavailable input not acknowledged: {spec['label']}",
            detail=(
                "This input slot was exposed by the architecture but the "
                "frozen test case marked the value unavailable. The memo "
                "should clearly acknowledge the limitation where material."
            ),
            expected={
                "item_id": item_id,
                "status": "unavailable",
            },
            observed="no clear acknowledgement found",
            location="llm_narrative",
        )

    return {
        "expected_unavailable_item_ids": expected_unavailable,
        "acknowledged_unavailable_item_ids": acknowledged,
        "missed_unavailable_item_ids": missed,
    }



def _check_rating_driver_discipline(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative: dict[str, Any],
) -> dict[str, Any]:
    groups = memo_context.get("rating_driver_groups", {}) or {}
    allowed_positive = {_normalise(x.get("label", "")) for x in groups.get("positive", []) if x.get("label")}
    allowed_negative = {_normalise(x.get("label", "")) for x in groups.get("negative", []) if x.get("label")}
    allowed_neutral = {_normalise(x.get("label", "")) for x in groups.get("neutral", []) if x.get("label")}

    negative_text = _normalise(_flatten(narrative.get("negative_rating_drivers")))
    positive_text = _normalise(_flatten(narrative.get("positive_rating_drivers")))
    commentary_text = _normalise(_flatten(narrative.get("rating_driver_commentary")))

    # Common failure: putting high leverage/liquidity in rating drivers when the actual rating feature is neutral.
    leverage_neutral = any("leverage" in x or "relative debt" in x for x in allowed_neutral)
    if leverage_neutral and _contains_any(negative_text + " " + commentary_text, ["debt / ebitda", "debt/ebitda", "high leverage", "elevated leverage", "net debt"]):
        _annotation(
            annotations,
            category="rating_driver_confusion",
            severity="medium",
            title="Financial leverage watchpoint treated as a negative rating driver",
            detail="The deterministic rating-driver group shows leverage/relative debt as neutral. Leverage can be a financial watchpoint or policy breach, but should not be described as a negative model driver unless the driver group supports it.",
            expected="Leverage belongs in financial watchpoints/policy unless rating_driver_groups.negative includes it.",
            observed=_snippet(_flatten(narrative), "leverage"),
            location="rating_driver_commentary/negative_rating_drivers",
        )

    # Another common failure: equity volatility listed as generic risk although it is supportive.
    if "equity volatility / market-implied risk" in allowed_positive:
        if _contains_any(positive_text, ["moderate risk", "manageable risk"]):
            _annotation(
                annotations,
                category="imprecise_driver_language",
                severity="low",
                title="Supportive equity-volatility diagnostic described imprecisely",
                detail="The diagnostic is supportive because equity volatility is favourable relative to the comparator, not because it is generically 'moderate risk'.",
                expected="Equity volatility / market-implied risk is supportive relative to median.",
                observed=_snippet(_flatten(narrative.get("positive_rating_drivers")), "volatility"),
                location="positive_rating_drivers",
            )

    return {
        "allowed_positive_driver_labels": sorted(allowed_positive),
        "allowed_negative_driver_labels": sorted(allowed_negative),
        "allowed_neutral_driver_labels": sorted(allowed_neutral),
    }


def _check_unsupported_claims(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative_text: str,
) -> dict[str, Any]:
    # For v0.1 these are conservative phrase checks. A later LLM judge can decide whether
    # each phrase is actually supported by business_summary or supplied facts.
    text_norm = _normalise(narrative_text)
    business_summary_norm = _normalise(_flatten(memo_context.get("borrower", {}).get("business_summary")))
    detected = []
    for phrase in UNSUPPORTED_PHRASES:
        if _normalise(phrase) in text_norm:
            # If exact phrase exists in supplied business summary, don't auto-flag it.
            if _normalise(phrase) in business_summary_norm:
                continue
            detected.append(phrase)
            _annotation(
                annotations,
                category="unsupported_claim_candidate",
                severity="info",
                title=f"Potential unsupported claim: '{phrase}'",
                detail="This phrase is a candidate for semantic support review. It does not reduce the deterministic score by itself.",
                expected="Use only supplied facts; avoid unsupported qualitative claims.",
                observed=_snippet(narrative_text, phrase),
                evidence={"phrase": phrase},
                location="narrative",
                scorer="deterministic_candidate_generator",
            )
    return {"potential_unsupported_phrases": detected, "candidates_scored": False}


def _check_approval_safety(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative_text: str,
) -> None:
    text = _normalise(narrative_text)
    policy_eval = memo_context.get("policy_evaluation") or {}
    requires_exception = bool(policy_eval.get("requires_exception_approval"))
    if not requires_exception:
        return

    risky_approval_patterns = [
        "recommend approval",
        "approval is recommended",
        "approve the request",
        "suitable for ordinary approval",
        "ordinary-course approval",
        "can be approved",
    ]
    if _contains_any(text, risky_approval_patterns) and not _contains_any(text, ["exception", "senior credit committee", "before any approval"]):
        _annotation(
            annotations,
            category="unsafe_recommendation",
            severity="critical",
            title="Approval language conflicts with exception-approval requirement",
            detail="The deterministic policy evaluation requires exception approval; the LLM should not recommend ordinary approval.",
            expected="Conditional exception-review language only.",
            observed="approval-like language without adequate exception framing",
            location="conclusion_recommendation",
        )


def _score(
    annotations: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    reference_coverage: dict[str, Any],
    positive_reasoning_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    reasoning, reasoning_annotations = _score_dimension(annotations, "reasoning")
    fidelity, fidelity_annotations = _score_dimension(annotations, "fidelity")
    tone, tone_annotations = _score_dimension(annotations, "tone")
    llm_performance = round((reasoning + fidelity + tone) / 3, 1)
    information_completeness = float(reference_coverage.get("score", 0.0))
    overall_memo_quality = round((information_completeness + llm_performance) / 2, 1)
    scored_annotations = [a for a in annotations if a.get("severity") != "info"
                          and a.get("category") not in {"unsupported_claim_candidate", "reference_coverage"}
                          and a.get("source_type", "llm") == "llm"]
    policy_diag = diagnostics.get("policy_detection", {})
    policy_scored = bool(policy_diag.get("policy_detection_scored"))
    expected_ids = policy_diag.get("expected_policy_ids", [])
    missed_ids = policy_diag.get("missed_policy_ids", [])
    policy_score = None if not policy_scored else (100 if not expected_ids else round(100 * (len(expected_ids)-len(missed_ids))/len(expected_ids),1))
    high_or_worse = [a for a in scored_annotations if a.get("severity") in {"critical","high"}]
    unsupported_candidates = [a for a in annotations if a.get("category") == "unsupported_claim_candidate"]
    factual = [a for a in scored_annotations if a.get("dimension") == "fidelity" and a.get("category") == "factual_error"]
    return {
        "information_completeness": information_completeness,
        "reasoning": reasoning, "fidelity": fidelity, "tone": tone,
        "llm_performance": llm_performance,
        "overall_memo_quality": overall_memo_quality,
        "dimension_method": "equal_weight_average",
        "llm_performance_formula": "(reasoning + fidelity + tone) / 3",
        "overall_memo_quality_formula": "(information_completeness + llm_performance) / 2",
        "reference_item_count": reference_coverage.get("total_reference_items",0),
        "supplied_reference_item_count": reference_coverage.get("supplied_reference_items",0),
        "missing_reference_item_count": reference_coverage.get("missing_reference_items",0),
        "dimension_issue_counts": {"reasoning": len(reasoning_annotations),
                                   "fidelity": len(fidelity_annotations),
                                   "tone": len(tone_annotations)},
        "positive_reasoning_finding_count": len(positive_reasoning_findings),
        "overall_score": llm_performance,
        "policy_detection_score": policy_score,
        "policy_detection_scored": policy_scored,
        "critical_or_high_issue_count": len(high_or_worse),
        "factual_error_count": len(factual),
        "unsupported_claim_count": 0,
        "unsupported_claim_candidate_count": len(unsupported_candidates),
        "scored_annotation_count": len(scored_annotations),
        "llm_annotation_count": len(annotations),
        "annotation_count": len(annotations),
    }

def annotate_credit_memo(
    memo_payload: dict[str, Any],
    document_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministically annotate a credit memo payload returned by create_credit_memo().

    Version 0.9.1 adds hidden-fact assertion checks, excludes industry/country rating anchors from borrower-rating extraction, strengthens policy-trigger classification and false-positive penalties, narrows generation-mechanics matching, and adds reasoning checks for explicitly unavailable inputs. It retains the v0.9 system-score architecture and the Reasoning, Fidelity and Tone dimensions. It is visibility-aware and distinguishes explicit borrower-rating
    assertions from policy thresholds. Policy-trigger scoring is disabled when the
    experiment supplied no policy evaluation, and phrase flags are unscored candidates.
    Human review or an LLM judge can then approve/edit/reject these draft annotations.
    """
    memo_context = memo_payload.get("memo_context") or {}
    narrative = memo_payload.get("narrative") or {}
    markdown = memo_payload.get("memo_markdown") or ""
    experiment_config = memo_payload.get("experiment_config") or {}

    source_manifest = _build_source_manifest(memo_payload)
    llm_narrative, narrative_text = _llm_narrative_for_scoring(narrative, source_manifest)
    combined_text = "\n".join([narrative_text, markdown])

    annotations: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    profile = _evaluation_profile(experiment_config)
    diagnostics["evaluation_profile"] = profile

    diagnostics["rating_and_pd"] = _check_core_rating_and_pd(annotations, memo_context, narrative_text, profile)
    _check_expected_loss(annotations, memo_context, narrative_text, profile)
    diagnostics["policy_detection"] = _check_policy_detection(annotations, memo_context, llm_narrative, profile)
    diagnostics["missing_information"] = _check_missing_information(
        annotations,
        memo_context,
        llm_narrative,
    )
    diagnostics["unavailable_inputs"] = _check_unavailable_inputs(
        annotations,
        memo_payload,
        llm_narrative,
        profile,
    )
    diagnostics["rating_driver_discipline"] = _check_rating_driver_discipline(
        annotations,
        memo_context,
        llm_narrative,
    )
    diagnostics["unsupported_claims"] = _check_unsupported_claims(annotations, memo_context, narrative_text)
    _check_approval_safety(annotations, memo_context, narrative_text)
    diagnostics["tone"] = _check_tone(annotations, llm_narrative, narrative_text)

    reference_coverage = _build_reference_coverage(memo_payload, memo_context, profile)
    diagnostics["reference_coverage"] = reference_coverage
    positive_reasoning_findings = _positive_reasoning_findings(annotations, diagnostics)
    diagnostics["positive_reasoning_findings"] = positive_reasoning_findings
    scores = _score(annotations, diagnostics, reference_coverage, positive_reasoning_findings)
    annotations_with_coverage = [_reference_coverage_annotation(reference_coverage), *annotations]

    return {
        "annotation_schema": ANNOTATION_SCHEMA,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_engine_version": ANNOTATION_ENGINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_metadata": {
            "experiment_id": experiment_config.get("experiment_id"),
            "context_mode": experiment_config.get("context_mode"),
            "policy_mode": experiment_config.get("policy_mode"),
            "prompt_mode": experiment_config.get("prompt_mode"),
            "model_tier": experiment_config.get("model_tier"),
            "model": experiment_config.get("model") or memo_payload.get("openai_model"),
            "narrative_source": memo_payload.get("narrative_source"),
        },
        "scores": scores,
        "diagnostics": diagnostics,
        "reference_coverage": reference_coverage,
        "source_manifest": source_manifest,
        "positive_reasoning_findings": positive_reasoning_findings,
        "annotations": annotations_with_coverage,
        "review_status": "draft_requires_human_review",
        "source_document": {
            "memo_id": memo_payload.get("memo_id"),
            "document_title": (document_map or {}).get("document_title"),
            "source_sha256": (document_map or {}).get("source_sha256"),
            "parser_version": (document_map or {}).get("parser_version"),
        },
    }


def annotation_config() -> dict[str, Any]:
    return {
        "annotation_schema": ANNOTATION_SCHEMA,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_engine_version": ANNOTATION_ENGINE_VERSION,
        "categories": [
            "reference_coverage",
            "factual_error",
            "policy_miss",
            "missing_information_miss",
            "rating_driver_confusion",
            "unsupported_claim",
            "unsafe_recommendation",
            "possible_false_positive",
            "overinclusive_policy_reference",
            "omission",
            "imprecise_driver_language",
            "excessive_caveating",
            "excessive_confidence",
            "repetition",
            "boilerplate",
            "generation_method_reference",
            "correct_detection",
            "correct_inference",
            "hidden_fact_assertion",
            "unavailable_input_miss",
        ],
        "llm_dimensions": list(LLM_DIMENSIONS),
        "severities": list(SEVERITY_WEIGHTS.keys()),
        "public_labels": ["Material Error ❌", "Important Limitation ⚠", "Strength ✓", "Review Note"],
        "notes": [
            "Deterministic annotations are a first-pass filter, not final judgement.",
            "Unsupported-claim detection is phrase-based in v0.1 and should be human reviewed.",
            "Policy scoring compares LLM policy prose against memo_context.policy_evaluation.",
            "Both LLM-evaluated and deterministic-policy modes receive completeness credit for the visible policy pack.",
            "Only llm_evaluated mode scores policy inference as LLM Reasoning.",
            "Policy references may be expressed using policy IDs or ordinary-language rule synonyms.",
            "Lowercase single-letter words are never treated as credit-rating assertions.",
            "Industry and country rating anchors are excluded from borrower-rating extraction.",
            "Any borrower rating or PD assertion is unsupported when the corresponding input is hidden by the architecture.",
            "Affirmative false policy triggers reduce Reasoning.",
            "Explicitly unavailable visible inputs should be acknowledged where material.",
            "Generation-mechanics checks use specific phrases rather than the standalone word prompt.",
            "Information completeness is count(supplied reference items) / count(all reference items).",
            "Each rule in the standard credit policy pack is a separate, equally weighted reference item.",
            "Policy-rule coverage is based on whether the full policy pack is exposed by the architecture, not whether a rule is triggered for the obligor.",
            "Reference coverage is determined before generation from architecture visibility and authoritative memo_context values.",
            "Reasoning measures conclusions, interpretation, prioritisation and risk awareness.",
            "Fidelity measures faithfulness to supplied evidence and avoidance of invention or distortion.",
            "Tone measures professional clarity, concision and confidence calibration.",
            "LLM performance is the equal-weight average of Reasoning, Fidelity and Tone.",
            "Overall memo quality is the simple average of Information Completeness and LLM Performance.",
            "Only content attributed to source_type=llm is eligible for LLM-performance deductions.",
            "The source manifest supports frontend highlighting of LLM-generated versus deterministic content.",
            "The legacy overall_score field remains the LLM performance score during transition.",
        ],
    }
