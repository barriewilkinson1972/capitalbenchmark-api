from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

ANNOTATION_SCHEMA = "credit_memo_annotation"
ANNOTATION_SCHEMA_VERSION = "1.0.0"
ANNOTATION_ENGINE_VERSION = "0.4.1"


# Phrases that explicitly state that a policy escalation does not apply.
# These prevent a simple synonym match from being counted as an affirmative
# policy reference.
POLICY_NEGATION_PATTERNS = {
    "CP-03": [
        "not requiring senior credit committee",
        "does not require senior credit committee",
        "not necessitating senior credit committee",
        "senior credit committee involvement is not required",
        "senior credit committee approval is not required",
        "no senior credit committee involvement",
        "no senior credit committee approval",
        "without senior credit committee involvement",
        "cp-03 does not apply",
        "cp-03 is not triggered",
        "cp-03 not triggered",
    ],
}

GENERIC_NEGATION_TERMS = [
    "not",
    "no",
    "does not",
    "do not",
    "did not",
    "is not",
    "are not",
    "was not",
    "were not",
    "without",
    "neither",
]

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

# Rating extraction deliberately avoids a trailing word-boundary after +/-.
# With a pattern like \bBBB\+\b, Python can fail to match "BBB+" because + is
# not a word character, and then separately match the substring "BBB". That creates
# false positive conflicts such as observed BBB vs expected BBB+.
RATING_RE = re.compile(
    r"(?<![A-Za-z])(?:AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|D)(?![A-Za-z+-])"
)

SINGLE_LETTER_RATINGS = {"A", "B", "C", "D"}


def _extract_rating_mentions(text: str) -> set[str]:
    """Extract plausible credit ratings from narrative text.

    Single-letter ratings are noisy in prose because they collide with ordinary
    words/articles. For now, keep them only when they appear close to explicit
    rating language.
    """
    mentions: set[str] = set()
    for match in RATING_RE.finditer(text or ""):
        rating = match.group(0)
        if rating in SINGLE_LETTER_RATINGS:
            window = (text[max(0, match.start() - 35): match.end() + 35] or "").lower()
            if not any(term in window for term in ["rating", "rated", "credit quality", "cb " ]):
                continue
        mentions.add(rating)
    return mentions



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



class DocumentIndex:
    """Convenience index over the nested document-map structure."""

    def __init__(self, document_map: dict[str, Any]):
        self.document_map = document_map or {}
        self.sections_by_id: dict[str, dict[str, Any]] = {}
        self.sections_by_type: dict[str, list[dict[str, Any]]] = {}
        self.blocks_by_id: dict[str, dict[str, Any]] = {}
        self.blocks_by_uuid: dict[str, dict[str, Any]] = {}
        self.blocks_by_section_type: dict[str, list[dict[str, Any]]] = {}
        self.blocks_in_order: list[dict[str, Any]] = []

        for section in self.document_map.get("sections", []) or []:
            section_id = str(section.get("section_id") or "")
            section_type = str(section.get("section_type") or "other")

            if section_id:
                self.sections_by_id[section_id] = section
            self.sections_by_type.setdefault(section_type, []).append(section)

            for block in section.get("blocks", []) or []:
                block_id = str(block.get("block_id") or "")
                block_uuid = str(block.get("block_uuid") or "")

                enriched = dict(block)
                enriched.setdefault("section_id", section_id)
                enriched.setdefault("section_type", section_type)
                enriched.setdefault("section_title", section.get("title"))

                if block_id:
                    self.blocks_by_id[block_id] = enriched
                if block_uuid:
                    self.blocks_by_uuid[block_uuid] = enriched

                self.blocks_by_section_type.setdefault(section_type, []).append(enriched)
                self.blocks_in_order.append(enriched)

        self.blocks_in_order.sort(
            key=lambda block: (
                int(block.get("document_order") or 0),
                str(block.get("block_id") or ""),
            )
        )

    def find_blocks(
        self,
        *,
        section_types: list[str] | None = None,
        search_terms: list[str] | None = None,
        require_all_terms: bool = False,
    ) -> list[dict[str, Any]]:
        candidates = self.blocks_in_order
        if section_types:
            allowed = set(section_types)
            candidates = [
                block for block in candidates
                if str(block.get("section_type") or "other") in allowed
            ]

        terms = [_normalise(term) for term in (search_terms or []) if _clean_text(term)]
        if not terms:
            return list(candidates)

        matches: list[dict[str, Any]] = []
        for block in candidates:
            haystack = _normalise(str(block.get("text") or ""))
            checks = [term in haystack for term in terms]
            if (require_all_terms and all(checks)) or (not require_all_terms and any(checks)):
                matches.append(block)
        return matches

    def first_block_in_section_types(self, section_types: list[str]) -> dict[str, Any] | None:
        matches = self.find_blocks(section_types=section_types)
        return matches[0] if matches else None


def _location_hint(
    *,
    target_preference: str,
    section_types: list[str] | None = None,
    search_terms: list[str] | None = None,
    missing_item: str | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    return {
        "target_preference": target_preference,
        "section_types": section_types or [],
        "search_terms": search_terms or [],
        "missing_item": missing_item,
        "policy_id": policy_id,
    }


def _block_location(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        return {
            "target_type": "unresolved",
            "resolution_status": "unresolved",
        }

    target_type = "block" if len(blocks) == 1 else "multi_block"
    orders = [int(block.get("document_order") or 0) for block in blocks]

    return {
        "target_type": target_type,
        "resolution_status": "resolved",
        "section_ids": list(dict.fromkeys(str(block.get("section_id") or "") for block in blocks)),
        "section_types": list(dict.fromkeys(str(block.get("section_type") or "other") for block in blocks)),
        "block_ids": [str(block.get("block_id") or "") for block in blocks],
        "block_uuids": [str(block.get("block_uuid") or "") for block in blocks],
        "document_order_start": min(orders) if orders else None,
        "document_order_end": max(orders) if orders else None,
        "target_excerpt": str(blocks[0].get("text") or "")[:240],
    }


def _resolve_location(
    hint: dict[str, Any] | None,
    document_index: DocumentIndex | None,
) -> dict[str, Any]:
    if not hint:
        return {
            "target_type": "unresolved",
            "resolution_status": "no_hint",
        }

    preference = str(hint.get("target_preference") or "matching_block")
    section_types = [str(x) for x in hint.get("section_types", []) or []]
    search_terms = [str(x) for x in hint.get("search_terms", []) or []]

    if preference == "missing_content":
        suggested = None
        if document_index:
            suggested = document_index.first_block_in_section_types(section_types)

        return {
            "target_type": "missing_content",
            "resolution_status": "resolved",
            "expected_section_types": section_types,
            "missing_item": hint.get("missing_item"),
            "suggested_insertion_after_block_id": (
                suggested.get("block_id") if suggested else None
            ),
        }

    if document_index is None:
        return {
            "target_type": "unresolved",
            "resolution_status": "document_map_not_supplied",
            "expected_section_types": section_types,
        }

    matches = document_index.find_blocks(
        section_types=section_types or None,
        search_terms=search_terms or None,
    )

    if matches:
        return _block_location(matches[:3])

    fallback = document_index.first_block_in_section_types(section_types) if section_types else None
    if fallback:
        location = _block_location([fallback])
        location["resolution_status"] = "resolved_by_section_fallback"
        return location

    return {
        "target_type": "unresolved",
        "resolution_status": "no_matching_block",
        "expected_section_types": section_types,
        "search_terms": search_terms,
    }


def _resolve_annotation_locations(
    annotations: list[dict[str, Any]],
    document_map: dict[str, Any] | None,
) -> dict[str, Any]:
    document_index = DocumentIndex(document_map) if document_map else None

    resolved = 0
    unresolved = 0
    missing_content = 0

    for annotation in annotations:
        location = _resolve_location(annotation.get("location_hint"), document_index)
        annotation["location"] = location

        if location.get("target_type") == "missing_content":
            missing_content += 1
            resolved += 1
        elif location.get("resolution_status", "").startswith("resolved"):
            resolved += 1
        else:
            unresolved += 1

    total = len(annotations)
    return {
        "annotation_count": total,
        "resolved_annotation_count": resolved,
        "unresolved_annotation_count": unresolved,
        "missing_content_count": missing_content,
        "location_resolution_score": 100.0 if total == 0 else round(100 * resolved / total, 1),
    }


def _validate_annotation_locations(
    annotations: list[dict[str, Any]],
    document_map: dict[str, Any] | None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not document_map:
        warnings.append({
            "code": "document_map_not_supplied",
            "message": "Locations could not be fully validated because no document map was supplied.",
        })
    else:
        index = DocumentIndex(document_map)
        for annotation in annotations:
            annotation_id = annotation.get("annotation_id")
            location = annotation.get("location") or {}
            for block_id in location.get("block_ids", []) or []:
                if block_id not in index.blocks_by_id:
                    errors.append({
                        "annotation_id": annotation_id,
                        "code": "unknown_block_id",
                        "block_id": block_id,
                    })
            for block_uuid in location.get("block_uuids", []) or []:
                if block_uuid and block_uuid not in index.blocks_by_uuid:
                    errors.append({
                        "annotation_id": annotation_id,
                        "code": "unknown_block_uuid",
                        "block_uuid": block_uuid,
                    })

    return {
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }

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
    location_hint: dict[str, Any] | None = None,
    confidence: float = 1.0,
    scorer: str = "deterministic",
) -> None:
    annotations.append(
        {
            "annotation_id": f"A{len(annotations) + 1:04d}",
            "category": category,
            "severity": severity,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "generated_finding": {
                "title": title,
                "detail": detail,
            },
            "published_finding": None,
            "title": title,
            "detail": detail,
            "expected": expected,
            "observed": observed,
            "evidence": evidence,
            "policy_id": policy_id,
            "location_hint": location_hint,
            "location": None,
            "scorer": scorer,
            "review": {
                "status": "unreviewed",
                "reviewer": None,
                "reviewed_at_utc": None,
                "decision": None,
                "notes": None,
            },
            "status": "draft",
            "public_label": _public_label(category, severity),
        }
    )


def _public_label(category: str, severity: str) -> str:
    if severity in {"critical", "high"}:
        return "Material Error ❌"
    if category in {"unsupported_claim", "omission", "missing_information_miss", "policy_miss", "rating_driver_confusion"}:
        return "Important Limitation ⚠"
    if category in {"strength", "correct_detection"}:
        return "Strength ✓"
    return "Review Note"


def _expected_policy_ids(memo_context: dict[str, Any]) -> list[str]:
    policy_eval = memo_context.get("policy_evaluation") or {}
    ids = policy_eval.get("triggered_policy_ids") or []
    return [str(x) for x in ids if x]


def _policy_term_matches(
    text_norm: str,
    policy_id: str,
    term: str,
) -> list[dict[str, Any]]:
    """Return polarity-aware matches for one policy term.

    Matching remains deliberately deterministic. A match is classified as
    negative when either:
      * a configured policy-specific negation phrase overlaps the context; or
      * a common negation term appears immediately before the matched term.

    The returned evidence is useful both for diagnostics and human review.
    """
    text_lower = _normalise(text_norm)
    term_norm = _normalise(term)
    if not term_norm:
        return []

    matches: list[dict[str, Any]] = []
    for match in re.finditer(re.escape(term_norm), text_lower):
        start, end = match.span()
        context_start = max(0, start - 90)
        context_end = min(len(text_lower), end + 70)
        context = text_lower[context_start:context_end].strip()

        policy_negation = next(
            (
                pattern
                for pattern in POLICY_NEGATION_PATTERNS.get(policy_id, [])
                if _normalise(pattern) in context
            ),
            None,
        )

        prefix = text_lower[max(0, start - 45):start]
        generic_negation = next(
            (
                negation
                for negation in GENERIC_NEGATION_TERMS
                if re.search(
                    rf"(?:^|[\s,;:()]){re.escape(negation)}(?:\s+\w+){{0,3}}\s*$",
                    prefix,
                )
            ),
            None,
        )

        polarity = "negative" if policy_negation or generic_negation else "positive"
        matches.append(
            {
                "policy_id": policy_id,
                "matched_term": term,
                "context": context,
                "polarity": polarity,
                "negation_pattern": policy_negation or generic_negation,
            }
        )

    return matches


def _policy_match_evidence(text_norm: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for policy_id, terms in POLICY_SYNONYMS.items():
        unique_terms = list(dict.fromkeys([policy_id, *terms]))
        for term in unique_terms:
            evidence.extend(_policy_term_matches(text_norm, policy_id, term))

    return evidence


def _visible_policy_ids(text_norm: str) -> set[str]:
    """Return only policies supported by an affirmative text match."""
    return {
        match["policy_id"]
        for match in _policy_match_evidence(text_norm)
        if match.get("polarity") == "positive"
    }


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
) -> None:
    rating = memo_context.get("capital_benchmark_rating", {})
    expected_rating = _clean_text(rating.get("cb_rating"))
    expected_pd = _safe_float(rating.get("cb_pd"))

    if expected_rating:
        mentioned_ratings = _extract_rating_mentions(narrative_text)
        wrong_ratings = sorted(r for r in mentioned_ratings if r != expected_rating)
        if wrong_ratings:
            _annotation(
                annotations,
                category="factual_error",
                severity="high",
                title="Conflicting credit rating mentioned in LLM prose",
                detail="The prose contains a rating that does not match the deterministic CB rating.",
                expected=expected_rating,
                observed=wrong_ratings,
                evidence={"all_ratings_mentioned": sorted(mentioned_ratings)},
                location_hint=_location_hint(target_preference="matching_block", section_types=["executive_summary", "rating_assessment"], search_terms=wrong_ratings),
            )
        elif expected_rating not in mentioned_ratings:
            _annotation(
                annotations,
                category="omission",
                severity="medium",
                title="CB rating not explicitly mentioned",
                detail="The deterministic CB rating should normally appear in the executive summary or rating assessment.",
                expected=expected_rating,
                observed="not found in narrative prose",
                location_hint=_location_hint(
                    target_preference="missing_content",
                    section_types=["executive_summary", "rating_assessment"],
                    missing_item="Capital Benchmark rating",
                ),
            )

    if expected_pd is not None:
        expected_pd_text = _format_percent(expected_pd)
        if expected_pd_text not in narrative_text:
            _annotation(
                annotations,
                category="omission",
                severity="low",
                title="CB PD not explicitly mentioned in expected format",
                detail="The deterministic CB PD should normally appear in the executive summary or rating assessment.",
                expected=expected_pd_text,
                observed="not found in narrative prose",
                location_hint=_location_hint(target_preference="missing_content", section_types=["executive_summary", "rating_assessment"], missing_item="Capital Benchmark PD"),
            )


def _money_value_is_mentioned(expected_value: float, text: str) -> bool:
    """Return True if a dollar amount appears in common memo formats.

    Accepts exact dollars ("$67,500"), compact thousands ("$67.5k"),
    and plain numeric variants. This avoids false omissions when different
    renderers/models use compact notation.
    """
    if expected_value is None:
        return False
    text_norm = (text or "").lower().replace(",", "")
    value = float(expected_value)
    candidates = {
        f"${value:,.0f}".lower().replace(",", ""),
        f"{value:,.0f}".lower().replace(",", ""),
        str(int(round(value))),
    }
    if abs(value) >= 1000:
        k = value / 1000
        candidates.update({
            f"${k:.1f}k".lower(),
            f"{k:.1f}k".lower(),
            f"${k:.0f}k".lower(),
            f"{k:.0f}k".lower(),
            f"${k:.1f} thousand".lower(),
            f"{k:.1f} thousand".lower(),
        })
    if abs(value) >= 1_000_000:
        m = value / 1_000_000
        candidates.update({
            f"${m:.1f}mn".lower(),
            f"{m:.1f}mn".lower(),
            f"${m:.1f} million".lower(),
            f"{m:.1f} million".lower(),
        })
    return any(candidate in text_norm for candidate in candidates if candidate)


def _check_expected_loss(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative_text: str,
) -> None:
    exposure = memo_context.get("exposure_analytics", {})
    expected_el = _safe_float(exposure.get("base_expected_loss_on_proposed_exposure"))
    if expected_el is None:
        return
    expected_text = _format_currency(expected_el)
    if not _money_value_is_mentioned(expected_el, narrative_text):
        _annotation(
            annotations,
            category="omission",
            severity="low",
            title="Expected loss not mentioned or not easy to verify",
            detail="Expected loss is deterministic and should be stated where exposure analytics are discussed.",
            expected=expected_text,
            observed="not found in narrative prose",
            location_hint=_location_hint(target_preference="missing_content", section_types=["executive_summary", "borrower_request_summary"], missing_item="expected loss"),
        )


def _check_policy_detection(
    annotations: list[dict[str, Any]],
    memo_context: dict[str, Any],
    narrative: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = set(_expected_policy_ids(memo_context))
    policy_text = _narrative_policy_text(narrative)
    policy_text_norm = _normalise(policy_text)
    policy_match_evidence = _policy_match_evidence(policy_text_norm)
    observed_ids = {
        match["policy_id"]
        for match in policy_match_evidence
        if match.get("polarity") == "positive"
    }
    positive_matches_by_policy = {
        policy_id: [
            match for match in policy_match_evidence
            if match.get("policy_id") == policy_id
            and match.get("polarity") == "positive"
        ]
        for policy_id in observed_ids
    }

    for policy_id in sorted(expected_ids):
        if policy_id not in observed_ids:
            _annotation(
                annotations,
                category="policy_miss",
                severity="high" if policy_id == "CP-03" else "medium",
                title=f"Expected policy finding not identified: {policy_id}",
                detail="The deterministic ground-truth policy evaluation triggered this rule, but the LLM policy prose did not clearly identify it.",
                expected=policy_id,
                observed=sorted(observed_ids),
                evidence=memo_context.get("policy_evaluation", {}).get("triggered_policies", []),
                policy_id=policy_id,
                location_hint=_location_hint(
                    target_preference="missing_content",
                    section_types=[
                        "policy_assessment",
                        "policy_breaches_triggers",
                        "required_policy_actions",
                        "policy_escalation_assessment",
                    ],
                    search_terms=POLICY_SYNONYMS.get(policy_id, [policy_id]),
                    missing_item=f"expected policy finding {policy_id}",
                    policy_id=policy_id,
                ),
            )

    # False positives are lower severity because a model can defensibly cite a lower threshold
    # such as CP-02 where CP-03 is the material severe trigger.
    for policy_id in sorted(observed_ids - expected_ids):
        if policy_id == "CP-02" and "CP-03" in expected_ids:
            severity = "info"
            detail = "The model cited CP-02 as a lesser-included leverage threshold while CP-03 is the material trigger. Treat as over-inclusive rather than wrong."
            category = "overinclusive_policy_reference"
        else:
            severity = "low"
            detail = "The model appears to cite a policy rule that is not present in the deterministic triggered-policy list."
            category = "possible_false_positive"
        _annotation(
            annotations,
            category=category,
            severity=severity,
            title=f"Additional policy reference not in deterministic triggered list: {policy_id}",
            detail=detail,
            expected=sorted(expected_ids),
            observed=policy_id,
            evidence={
                "policy_matches": positive_matches_by_policy.get(policy_id, []),
            },
            policy_id=policy_id,
            location_hint=_location_hint(
                target_preference="matching_block",
                section_types=[
                    "policy_assessment",
                    "policy_breaches_triggers",
                    "required_policy_actions",
                    "policy_escalation_assessment",
                ],
                search_terms=[
                    match.get("matched_term")
                    for match in positive_matches_by_policy.get(policy_id, [])
                    if match.get("matched_term")
                ] or POLICY_SYNONYMS.get(policy_id, [policy_id]),
                policy_id=policy_id,
            ),
        )

    # Escalation checks.
    policy_eval = memo_context.get("policy_evaluation") or {}
    if policy_eval.get("requires_senior_credit_committee"):
        if not _contains_any(policy_text_norm, ["senior credit committee", "exception approval", "exception-approval"]):
            _annotation(
                annotations,
                category="policy_miss",
                severity="high",
                title="Senior credit committee escalation not clearly stated",
                detail="The deterministic policy evaluation requires senior credit committee exception approval.",
                expected="senior credit committee exception approval",
                observed="not clearly found",
                policy_id="CP-03",
                location_hint=_location_hint(target_preference="matching_block", section_types=["policy_escalation_assessment"], search_terms=["senior credit committee", "exception approval"], policy_id="CP-03"),
            )

    return {
        "expected_policy_ids": sorted(expected_ids),
        "observed_policy_ids": sorted(observed_ids),
        "missed_policy_ids": sorted(expected_ids - observed_ids),
        "additional_policy_ids": sorted(observed_ids - expected_ids),
        "policy_match_evidence": policy_match_evidence,
        "negative_policy_matches": [
            match
            for match in policy_match_evidence
            if match.get("polarity") == "negative"
        ],
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
                location_hint=_location_hint(
                    target_preference="missing_content",
                    section_types=[
                        "policy_missing_information",
                        "relationship_manager_questions",
                    ],
                    missing_item=item,
                ),
                confidence=0.80,
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
            location_hint=_location_hint(target_preference="matching_block", section_types=["rating_driver_commentary", "negative_rating_drivers"], search_terms=["leverage", "debt / ebitda", "net debt"]),
            confidence=0.75,
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
                location_hint=_location_hint(target_preference="matching_block", section_types=["positive_rating_drivers"], search_terms=["volatility", "market-implied risk"]),
                confidence=0.75,
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
                category="unsupported_claim",
                severity="medium" if phrase in {"market leader", "leading provider", "operational stability", "approval is recommended"} else "low",
                title=f"Potential unsupported claim: '{phrase}'",
                detail="This phrase is often unsupported unless explicitly present in the supplied context. Review before publication.",
                expected="Use only supplied facts; avoid unsupported qualitative claims.",
                observed=_snippet(narrative_text, phrase),
                evidence={"phrase": phrase},
                location_hint=_location_hint(target_preference="matching_block", search_terms=[phrase]),
                confidence=0.55,
            )
    return {"potential_unsupported_phrases": detected}


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
            location_hint=_location_hint(target_preference="matching_block", section_types=["conclusion_recommendation"], search_terms=risky_approval_patterns),
        )


def _score(annotations: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    penalty = sum(SEVERITY_WEIGHTS.get(a.get("severity", "low"), 2) for a in annotations if a.get("severity") != "info")
    overall = max(0, 100 - penalty)

    expected_ids = diagnostics.get("policy_detection", {}).get("expected_policy_ids", [])
    missed_ids = diagnostics.get("policy_detection", {}).get("missed_policy_ids", [])
    policy_score = 100 if not expected_ids else round(100 * (len(expected_ids) - len(missed_ids)) / len(expected_ids), 1)

    expected_missing = diagnostics.get("missing_information", {}).get("expected_missing_information", [])
    missed_missing = diagnostics.get("missing_information", {}).get("missed_missing_information", [])
    missing_score = 100 if not expected_missing else round(100 * (len(expected_missing) - len(missed_missing)) / len(expected_missing), 1)

    high_or_worse = [a for a in annotations if a.get("severity") in {"critical", "high"}]
    unsupported = [a for a in annotations if a.get("category") == "unsupported_claim"]
    factual = [a for a in annotations if a.get("category") == "factual_error"]

    return {
        "overall_score": overall,
        "policy_detection_score": policy_score,
        "missing_information_detection_score": missing_score,
        "critical_or_high_issue_count": len(high_or_worse),
        "factual_error_count": len(factual),
        "unsupported_claim_count": len(unsupported),
        "annotation_count": len(annotations),
    }


def annotate_credit_memo(
    memo_payload: dict[str, Any],
    document_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministically annotate a credit memo payload returned by create_credit_memo().

    This v0.1 module is deliberately conservative: it is a first-pass filter for
    obvious factual conflicts, missed deterministic policy findings, missing-information
    misses, rating-driver/watchpoint confusion, and likely unsupported phrases.
    Human review or an LLM judge can then approve/edit/reject these draft annotations.
    """
    memo_context = memo_payload.get("memo_context") or {}
    narrative = memo_payload.get("narrative") or {}
    markdown = memo_payload.get("memo_markdown") or ""
    experiment_config = memo_payload.get("experiment_config") or {}

    narrative_text = _flatten(narrative)
    visible_text = markdown or narrative_text

    annotations: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    _check_core_rating_and_pd(annotations, memo_context, visible_text)
    _check_expected_loss(annotations, memo_context, visible_text)
    diagnostics["policy_detection"] = _check_policy_detection(annotations, memo_context, narrative)
    diagnostics["missing_information"] = _check_missing_information(annotations, memo_context, narrative)
    diagnostics["rating_driver_discipline"] = _check_rating_driver_discipline(annotations, memo_context, narrative)
    diagnostics["unsupported_claims"] = _check_unsupported_claims(annotations, memo_context, visible_text)
    _check_approval_safety(annotations, memo_context, visible_text)

    diagnostics["location_resolution"] = _resolve_annotation_locations(annotations, document_map)
    validation = _validate_annotation_locations(annotations, document_map)
    scores = _score(annotations, diagnostics)
    scores["location_resolution_score"] = diagnostics["location_resolution"]["location_resolution_score"]

    return {
        "annotation_schema": ANNOTATION_SCHEMA,
        "annotation_schema": ANNOTATION_SCHEMA,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_engine_version": ANNOTATION_ENGINE_VERSION,
        "annotation_engine_version": ANNOTATION_ENGINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_document": {
            "memo_id": (document_map or {}).get("memo_id"),
            "document_title": (document_map or {}).get("document_title"),
            "source_sha256": (document_map or {}).get("source_sha256"),
            "parser_version": (document_map or {}).get("parser_version"),
        },
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
        "validation": validation,
        "annotations": annotations,
        "review_status": "draft_requires_human_review",
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
