from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

ANNOTATION_SCHEMA = "credit_memo_annotation"
ANNOTATION_SCHEMA_VERSION = "1.0.0"
ANNOTATION_ENGINE_VERSION = "0.6.0"

SEVERITY_WEIGHTS = {
    "critical": 20,
    "high": 10,
    "medium": 5,
    "low": 2,
    "info": 0,
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
    "minimum rating", "threshold", "below", "above", "or below", "or better",
    "policy", "trigger", "category", "single-b", "single b", "rating scale",
    "agency rating", "external rating", "benchmarking",
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
) -> None:
    annotations.append(
        {
            "annotation_id": f"A{len(annotations) + 1:04d}",
            "category": category,
            "severity": severity,
            "title": title,
            "detail": detail,
            "expected": expected,
            "observed": observed,
            "evidence": evidence,
            "policy_id": policy_id,
            "location": location,
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


def _evaluation_profile(experiment_config: dict[str, Any]) -> dict[str, Any]:
    context_mode = _clean_text(experiment_config.get("context_mode")).lower()
    policy_mode = _clean_text(experiment_config.get("policy_mode")).lower()

    rating_expected = context_mode in {"rating_only", "rating_and_financials", "full"}
    financials_visible = context_mode in {"financials_only", "rating_and_financials", "full"}
    policy_trigger_detection_expected = policy_mode in {"llm_evaluated", "deterministic_evaluated"}

    return {
        "context_mode": context_mode or None,
        "policy_mode": policy_mode or None,
        "rating_visible": rating_expected,
        "rating_expected": rating_expected,
        "pd_expected": rating_expected,
        "financials_visible": financials_visible,
        "expected_loss_expected": context_mode == "full",
        "policy_trigger_detection_expected": policy_trigger_detection_expected,
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
                rating = match.group("rating").upper()
                start, end = match.span("rating")
                local = _normalise(sentence[max(0, start - 55): min(len(sentence), end + 55)])
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


def _classify_policy_reference(sentence: str, policy_id: str) -> str:
    norm = _normalise(sentence)
    negative_patterns = [
        "not triggered", "not breached", "no breach", "satisfied", "compliant",
        "no issue", "does not trigger",
    ]
    not_assessable_patterns = [
        "not assessable", "cannot be assessed", "unable to assess", "not provided",
        "not available", "insufficient information", "cannot be determined",
    ]
    conditional_patterns = [
        "if ", "would require", "would trigger", "subject to", "once provided",
        "upon receipt", "if evidenced", "to assess", "to test",
    ]
    triggered_patterns = [
        "breach", "breached", "triggered", "not met", "below the minimum",
        "exceeds", "requires exception", "exception approval required",
        "must be obtained", "is missing", "requires enhanced review",
    ]

    if any(x in norm for x in not_assessable_patterns):
        return "not_assessable"
    if any(x in norm for x in negative_patterns):
        return "not_triggered"
    if any(x in norm for x in conditional_patterns):
        return "conditional"
    if any(x in norm for x in triggered_patterns):
        return "triggered"
    return "mentioned"


def _policy_references(text: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for sentence in _sentence_windows(text):
        explicit = set(re.findall(r"\bCP-\d{2}\b", sentence.upper()))
        for policy_id in sorted(explicit):
            refs.append({
                "policy_id": policy_id,
                "status": _classify_policy_reference(sentence, policy_id),
                "sentence": sentence,
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
    expected_rating = _clean_text(rating.get("cb_rating")).upper()
    expected_pd = _safe_float(rating.get("cb_pd"))
    assertions = _extract_borrower_rating_assertions(narrative_text)
    asserted_ratings = sorted({x["rating"] for x in assertions})

    if expected_rating:
        wrong_ratings = sorted(r for r in asserted_ratings if r != expected_rating)
        if wrong_ratings:
            _annotation(
                annotations,
                category="factual_error",
                severity="high",
                title="Conflicting borrower credit rating asserted in LLM prose",
                detail="The prose explicitly asserts a borrower/internal rating that does not match the deterministic CB rating. Policy thresholds and rating-category references are excluded.",
                expected=expected_rating,
                observed=wrong_ratings,
                evidence={"borrower_rating_assertions": assertions},
                location="narrative",
            )
        elif profile.get("rating_expected") and expected_rating not in asserted_ratings:
            _annotation(
                annotations,
                category="omission",
                severity="medium",
                title="CB rating not explicitly mentioned",
                detail="The CB rating was visible in this experimental context and should normally appear in the executive summary or rating assessment.",
                expected=expected_rating,
                observed="no borrower-rating assertion found",
                evidence={"borrower_rating_assertions": assertions},
                location="narrative",
            )

    if expected_pd is not None and profile.get("pd_expected"):
        expected_pd_text = _format_percent(expected_pd)
        pd_variants = {
            expected_pd_text,
            expected_pd_text.replace(".00%", "%"),
            f"{expected_pd * 100:.1f}%",
        }
        if not any(x in narrative_text for x in pd_variants):
            _annotation(
                annotations,
                category="omission",
                severity="low",
                title="CB PD not explicitly mentioned",
                detail="The CB PD was visible in this experimental context and should normally appear in the executive summary or rating assessment.",
                expected=expected_pd_text,
                observed="not found in narrative prose",
                location="narrative",
            )

    return {
        "expected_rating": expected_rating or None,
        "borrower_rating_assertions": assertions,
        "asserted_borrower_ratings": asserted_ratings,
        "rating_scored": bool(profile.get("rating_expected")),
        "pd_scored": bool(profile.get("pd_expected")),
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
                detail = "The memo affirmatively cited CP-02 alongside the more severe CP-03 trigger. This is treated as over-inclusive rather than a material error."
            else:
                category = "possible_false_positive"
                severity = "info"
                detail = "The memo appears to state this policy as triggered, but it is not in the deterministic triggered-policy list. Review the local context before publication."
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


def _score(annotations: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    scored_annotations = [
        a for a in annotations
        if a.get("severity") != "info" and a.get("category") != "unsupported_claim_candidate"
    ]
    penalty = sum(SEVERITY_WEIGHTS.get(a.get("severity", "low"), 2) for a in scored_annotations)
    overall = max(0, 100 - penalty)

    policy_diag = diagnostics.get("policy_detection", {})
    policy_scored = bool(policy_diag.get("policy_detection_scored"))
    expected_ids = policy_diag.get("expected_policy_ids", [])
    missed_ids = policy_diag.get("missed_policy_ids", [])
    if not policy_scored:
        policy_score = None
    else:
        policy_score = 100 if not expected_ids else round(100 * (len(expected_ids) - len(missed_ids)) / len(expected_ids), 1)

    expected_missing = diagnostics.get("missing_information", {}).get("expected_missing_information", [])
    missed_missing = diagnostics.get("missing_information", {}).get("missed_missing_information", [])
    missing_score = 100 if not expected_missing else round(100 * (len(expected_missing) - len(missed_missing)) / len(expected_missing), 1)

    high_or_worse = [a for a in scored_annotations if a.get("severity") in {"critical", "high"}]
    unsupported_candidates = [a for a in annotations if a.get("category") == "unsupported_claim_candidate"]
    factual = [a for a in scored_annotations if a.get("category") == "factual_error"]

    return {
        "overall_score": overall,
        "policy_detection_score": policy_score,
        "policy_detection_scored": policy_scored,
        "missing_information_detection_score": missing_score,
        "critical_or_high_issue_count": len(high_or_worse),
        "factual_error_count": len(factual),
        "unsupported_claim_count": 0,
        "unsupported_claim_candidate_count": len(unsupported_candidates),
        "scored_annotation_count": len(scored_annotations),
        "annotation_count": len(annotations),
    }

def annotate_credit_memo(
    memo_payload: dict[str, Any],
    document_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministically annotate a credit memo payload returned by create_credit_memo().

    Version 0.6.0 is visibility-aware and distinguishes explicit borrower-rating
    assertions from policy thresholds. Policy-trigger scoring is disabled when the
    experiment supplied no policy evaluation, and phrase flags are unscored candidates.
    Human review or an LLM judge can then approve/edit/reject these draft annotations.
    """
    memo_context = memo_payload.get("memo_context") or {}
    narrative = memo_payload.get("narrative") or {}
    markdown = memo_payload.get("memo_markdown") or ""
    experiment_config = memo_payload.get("experiment_config") or {}

    narrative_text = _flatten(narrative)
    combined_text = "\n".join([narrative_text, markdown])

    annotations: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    profile = _evaluation_profile(experiment_config)
    diagnostics["evaluation_profile"] = profile

    diagnostics["rating_and_pd"] = _check_core_rating_and_pd(annotations, memo_context, narrative_text, profile)
    _check_expected_loss(annotations, memo_context, narrative_text, profile)
    diagnostics["policy_detection"] = _check_policy_detection(annotations, memo_context, narrative, profile)
    diagnostics["missing_information"] = _check_missing_information(annotations, memo_context, narrative)
    diagnostics["rating_driver_discipline"] = _check_rating_driver_discipline(annotations, memo_context, narrative)
    diagnostics["unsupported_claims"] = _check_unsupported_claims(annotations, memo_context, narrative_text)
    _check_approval_safety(annotations, memo_context, narrative_text)

    scores = _score(annotations, diagnostics)

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
        "annotations": annotations,
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
        ],
        "severities": list(SEVERITY_WEIGHTS.keys()),
        "public_labels": ["Material Error ❌", "Important Limitation ⚠", "Strength ✓", "Review Note"],
        "notes": [
            "Deterministic annotations are a first-pass filter, not final judgement.",
            "Unsupported-claim detection is phrase-based in v0.1 and should be human reviewed.",
            "Policy scoring compares LLM policy prose against memo_context.policy_evaluation.",
        ],
    }
