from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any


DEFAULT_REPORT_MODEL = os.getenv("OPENAI_SCENARIO_REPORT_MODEL", "gpt-4o-mini")


SCENARIO_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "scenario_label": {"type": "string"},
        "executive_summary": {"type": "string"},
        "scenario_interpretation": {"type": "string"},
        "portfolio_impact": {"type": "string"},
        "industry_concentration": {"type": "string"},
        "obligor_concentration": {"type": "string"},
        "key_risk_drivers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "management_takeaways": {
            "type": "array",
            "items": {"type": "string"},
        },
        "limitations": {"type": "string"},
    },
    "required": [
        "title",
        "scenario_label",
        "executive_summary",
        "scenario_interpretation",
        "portfolio_impact",
        "industry_concentration",
        "obligor_concentration",
        "key_risk_drivers",
        "management_takeaways",
        "limitations",
    ],
    "additionalProperties": False,
}


SCENARIO_REPORT_INSTRUCTIONS = """
You are a senior credit portfolio risk analyst writing for banks and credit risk teams.

Write a concise institutional scenario report using only the supplied JSON data.
Do not invent numbers. Do not recalculate figures. Do not mention fields that are absent.
Narrate the economic meaning of the scenario, the main loss drivers, concentrations,
and risk-management implications.

The backend owns all numbers and tables. Your job is to provide the prose sections
required by the JSON schema. Use a professional but readable tone.
Avoid investment advice, trading recommendations, and false precision.

Scenario factor semantics:
- Negative market values mean broad market stress / risk-off conditions.
- Negative technology values mean technology-sector stress or unwind.
- Positive commodity values mean a commodity price shock / input-cost shock. Do not
  describe this as favorable for the portfolio unless the supplied factor loadings
  and loss data support that interpretation.

Style rules:
- Use compact units in prose, such as "$262.6bn" and "1.69%", rather than long
  unformatted numbers.
- Avoid repeating every table number in prose. Summarise the implication.
- If a factor sign looks directionally mixed across industries, describe the
  industry-specific sensitivity rather than calling the factor universally good
  or bad.
""".strip()


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _sort_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: _safe_float(row.get(field), 0.0) or 0.0,
        reverse=True,
    )


def _sum_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum((_safe_float(row.get(field), 0.0) or 0.0) for row in rows))


def _weighted_average(rows: list[dict[str, Any]], value_field: str, weight_field: str) -> float | None:
    denominator = _sum_field(rows, weight_field)
    if denominator <= 0:
        return None
    numerator = sum(
        (_safe_float(row.get(value_field), 0.0) or 0.0)
        * (_safe_float(row.get(weight_field), 0.0) or 0.0)
        for row in rows
    )
    return float(numerator / denominator)


def _top_share(rows: list[dict[str, Any]], n: int, field: str) -> float | None:
    total = _sum_field(rows, field)
    if total <= 0:
        return None
    return _sum_field(_sort_rows(rows, field)[:n], field) / total


def _compact_industry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "industry": row.get("industry"),
        "obligors": _safe_int(row.get("obligors"), 0),
        "ead": _safe_float(row.get("ead"), 0.0),
        "base_pd": _safe_float(row.get("base_pd")),
        "stressed_pd": _safe_float(row.get("stressed_pd")),
        "stressed_expected_loss": _safe_float(row.get("stressed_expected_loss"), 0.0),
        "pd_multiple": _safe_float(row.get("pd_multiple")),
        "rho_Market": _safe_float(row.get("rho_Market")),
        "rho_Technology": _safe_float(row.get("rho_Technology")),
        "rho_Commodity": _safe_float(row.get("rho_Commodity")),
    }


def _compact_obligor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": row.get("symbol"),
        "company_name": row.get("company_name"),
        "agency_rating": row.get("Agency Rating") or row.get("agency_rating"),
        "cb_rating": row.get("cb_rating"),
        "industry": row.get("industry"),
        "sector": row.get("sector"),
        "country": row.get("country"),
        "ead": _safe_float(row.get("ead"), 0.0),
        "base_pd": _safe_float(row.get("base_pd")),
        "stressed_pd": _safe_float(row.get("stressed_pd")),
        "stressed_expected_loss": _safe_float(row.get("stressed_expected_loss"), 0.0),
        "pd_multiple": _safe_float(row.get("pd_multiple")),
    }


def build_report_context(
    stress_result: dict[str, Any],
    scenario: dict[str, Any],
    top_industries_n: int = 20,
    top_obligors_n: int = 20,
) -> dict[str, Any]:
    summary = dict(stress_result.get("summary", {}))
    scenario_from_result = dict(stress_result.get("scenario", {}))

    industries = [
        _compact_industry(row)
        for row in stress_result.get("top_industries", [])
        if isinstance(row, dict)
    ]
    obligors = [
        _compact_obligor(row)
        for row in stress_result.get("top_obligors", [])
        if isinstance(row, dict)
    ]

    industries = _sort_rows(industries, "stressed_expected_loss")
    obligors = _sort_rows(obligors, "stressed_expected_loss")

    industry_total_ead = _sum_field(industries, "ead")
    industry_stressed_el = _sum_field(industries, "stressed_expected_loss")

    weighted_base_pd = _weighted_average(industries, "base_pd", "ead")
    weighted_stressed_pd = _weighted_average(industries, "stressed_pd", "ead")

    base_expected_loss = _safe_float(summary.get("base_expected_loss"))
    if base_expected_loss is None and industry_total_ead > 0 and weighted_base_pd is not None:
        lgd = _safe_float(summary.get("lgd"), 0.45) or 0.45
        base_expected_loss = industry_total_ead * weighted_base_pd * lgd

    expected_loss_multiple = _safe_float(summary.get("expected_loss_multiple"))
    if expected_loss_multiple is None and base_expected_loss and base_expected_loss > 0:
        expected_loss_multiple = industry_stressed_el / base_expected_loss

    merged_scenario = {
        **scenario_from_result,
        **scenario,
    }

    context = {
        "report_version": "scenario_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "factor_semantics": {
            "market": {
                "negative": "broad market stress / risk-off conditions",
                "positive": "supportive market conditions",
            },
            "technology": {
                "negative": "technology-sector stress or valuation unwind",
                "positive": "technology-sector boom or supportive technology conditions",
            },
            "commodity": {
                "negative": "commodity price relief / lower input-cost pressure",
                "positive": "commodity price shock / higher input-cost pressure",
            },
        },
        "scenario": merged_scenario,
        "portfolio_summary": {
            "total_ead": _safe_float(summary.get("total_ead"), industry_total_ead),
            "base_expected_loss": base_expected_loss,
            "stressed_expected_loss": _safe_float(
                summary.get("stressed_expected_loss"),
                industry_stressed_el,
            ),
            "stressed_loss_rate": (
                industry_stressed_el / industry_total_ead
                if industry_total_ead > 0
                else _safe_float(summary.get("stressed_loss_rate"))
            ),
            "expected_loss_multiple": expected_loss_multiple,
            "weighted_base_pd": _safe_float(
                summary.get("base_portfolio_pd"),
                weighted_base_pd,
            ),
            "weighted_stressed_pd": _safe_float(
                summary.get("stressed_portfolio_pd"),
                weighted_stressed_pd,
            ),
            "scenario_tail_probability": _safe_float(
                summary.get("scenario_tail_probability")
            ),
            "scenario_tail_odds": _safe_float(summary.get("scenario_tail_odds")),
        },
        "concentration": {
            "top_5_industry_loss_share": _top_share(industries, 5, "stressed_expected_loss"),
            "top_10_industry_loss_share": _top_share(industries, 10, "stressed_expected_loss"),
            "top_20_industry_loss_share": _top_share(industries, 20, "stressed_expected_loss"),
            "top_10_obligor_loss_share_of_export": _top_share(
                obligors,
                10,
                "stressed_expected_loss",
            ),
            "top_20_obligor_loss_share_of_export": _top_share(
                obligors,
                20,
                "stressed_expected_loss",
            ),
            "industry_rows_available": len(industries),
            "obligor_rows_available": len(obligors),
        },
        "top_industries": industries[:top_industries_n],
        "top_obligors": obligors[:top_obligors_n],
    }

    return _json_safe(context)


def generate_scenario_narrative(
    report_context: dict[str, Any],
    model: str = DEFAULT_REPORT_MODEL,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=SCENARIO_REPORT_INSTRUCTIONS,
        input=json.dumps(report_context, separators=(",", ":"), ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "scenario_report_narrative",
                "strict": True,
                "schema": SCENARIO_REPORT_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


def fallback_scenario_narrative(report_context: dict[str, Any]) -> dict[str, Any]:
    scenario = report_context.get("scenario", {})
    summary = report_context.get("portfolio_summary", {})
    industries = report_context.get("top_industries", [])
    obligors = report_context.get("top_obligors", [])

    market = scenario.get("market")
    technology = scenario.get("technology")
    commodity = scenario.get("commodity")
    top_industry = industries[0].get("industry") if industries else "the leading stressed industries"
    top_obligor = obligors[0].get("company_name") if obligors else "the leading stressed obligors"

    return {
        "title": "Capital Benchmark Scenario Report",
        "scenario_label": (
            f"Market {market:+g}, technology {technology:+g}, commodity {commodity:+g}"
            if all(v is not None for v in [market, technology, commodity])
            else "Custom stress scenario"
        ),
        "executive_summary": (
            "This scenario produces a material deterioration in portfolio credit quality. "
            "The strongest loss contributions are concentrated in cyclical, technology-linked, "
            "and commodity-sensitive industries."
        ),
        "scenario_interpretation": (
            "The scenario combines broad market stress with a technology-factor downturn and "
            "a commodity shock. This combination weakens refinancing conditions, pressures "
            "cyclical earnings, and increases default risk in sectors with high factor loadings."
        ),
        "portfolio_impact": (
            "The portfolio impact is visible through higher stressed PDs and increased expected "
            f"loss. Stressed expected loss is {format_currency(summary.get('stressed_expected_loss'))}, "
            f"or {format_percent(summary.get('stressed_loss_rate'))} of EAD."
        ),
        "industry_concentration": (
            f"Industry losses are led by {top_industry}. The top industries account for a "
            "disproportionate share of stressed expected loss relative to their EAD share."
        ),
        "obligor_concentration": (
            f"Name-level concentration is led by {top_obligor}. The largest obligors should be "
            "reviewed individually because single-name EAD can materially influence scenario loss."
        ),
        "key_risk_drivers": [
            "Broad market stress increases risk across high-beta corporate sectors.",
            "Technology stress affects semiconductors and other technology-linked industries.",
            "Commodity stress affects sectors with adverse energy-cost or input-cost sensitivity.",
        ],
        "management_takeaways": [
            "Review the largest industry and obligor contributors before using the report externally.",
            "Compare stressed loss concentration with portfolio limits and risk appetite.",
            "Use the output as scenario analytics, not as a standalone capital requirement.",
        ],
        "limitations": (
            "This report is generated from model outputs and does not incorporate external news, "
            "management overlays, covenant analysis, liquidity analysis, or borrower-specific judgement."
        ),
    }


def format_currency(value: Any) -> str:
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


def format_percent(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.2f}%"


def format_multiple(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.1f}x"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _portfolio_summary_table(report_context: dict[str, Any]) -> str:
    summary = report_context.get("portfolio_summary", {})
    rows = [
        ["Total EAD", format_currency(summary.get("total_ead"))],
        ["Weighted base PD", format_percent(summary.get("weighted_base_pd"))],
        ["Weighted stressed PD", format_percent(summary.get("weighted_stressed_pd"))],
        ["Base expected loss", format_currency(summary.get("base_expected_loss"))],
        ["Stressed expected loss", format_currency(summary.get("stressed_expected_loss"))],
        ["Stressed expected loss / EAD", format_percent(summary.get("stressed_loss_rate"))],
        ["Expected loss multiple", format_multiple(summary.get("expected_loss_multiple"))],
    ]
    if summary.get("scenario_tail_probability") is not None:
        rows.append([
            "Scenario tail probability",
            format_percent(summary.get("scenario_tail_probability")),
        ])
    if summary.get("scenario_tail_odds") is not None:
        rows.append([
            "Scenario tail odds",
            f"1 in {summary.get('scenario_tail_odds'):,.0f}",
        ])
    return _markdown_table(["Metric", "Value"], rows)


def _industry_table(report_context: dict[str, Any], limit: int = 10) -> str:
    rows = []
    for row in report_context.get("top_industries", [])[:limit]:
        rows.append([
            str(row.get("industry") or ""),
            str(row.get("obligors") or ""),
            format_currency(row.get("ead")),
            format_percent(row.get("stressed_pd")),
            format_currency(row.get("stressed_expected_loss")),
            format_multiple(row.get("pd_multiple")),
        ])
    return _markdown_table(
        ["Industry", "Obligors", "EAD", "Stressed PD", "Stressed EL", "PD Multiple"],
        rows,
    )


def _obligor_table(report_context: dict[str, Any], limit: int = 10) -> str:
    rows = []
    for row in report_context.get("top_obligors", [])[:limit]:
        rows.append([
            str(row.get("company_name") or row.get("symbol") or ""),
            str(row.get("industry") or ""),
            str(row.get("agency_rating") or row.get("cb_rating") or ""),
            format_currency(row.get("ead")),
            format_percent(row.get("stressed_pd")),
            format_currency(row.get("stressed_expected_loss")),
        ])
    return _markdown_table(
        ["Obligor", "Industry", "Rating", "EAD", "Stressed PD", "Stressed EL"],
        rows,
    )


def render_markdown_report(
    report_context: dict[str, Any],
    narrative: dict[str, Any],
) -> str:
    drivers = "\n".join(
        f"- {item}" for item in narrative.get("key_risk_drivers", [])
    )
    takeaways = "\n".join(
        f"- {item}" for item in narrative.get("management_takeaways", [])
    )

    return f"""# {narrative.get("title", "Capital Benchmark Scenario Report")}

**Scenario:** {narrative.get("scenario_label", "Custom stress scenario")}

## Executive Summary

{narrative.get("executive_summary", "")}

## Scenario Interpretation

{narrative.get("scenario_interpretation", "")}

## Portfolio Impact

{narrative.get("portfolio_impact", "")}

{_portfolio_summary_table(report_context)}

## Industry Concentration

{narrative.get("industry_concentration", "")}

{_industry_table(report_context)}

## Obligor Concentration

{narrative.get("obligor_concentration", "")}

{_obligor_table(report_context)}

## Key Risk Drivers

{drivers}

## Management Takeaways

{takeaways}

## Limitations

{narrative.get("limitations", "")}
""".strip()


def create_scenario_report(
    stress_result: dict[str, Any],
    scenario: dict[str, Any],
    use_openai: bool = True,
    model: str = DEFAULT_REPORT_MODEL,
    top_industries_n: int = 20,
    top_obligors_n: int = 20,
    require_openai: bool = False,
) -> dict[str, Any]:
    report_context = build_report_context(
        stress_result=stress_result,
        scenario=scenario,
        top_industries_n=top_industries_n,
        top_obligors_n=top_obligors_n,
    )

    openai_api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    fallback_reason = None

    if use_openai and openai_api_key_present:
        try:
            narrative = generate_scenario_narrative(report_context, model=model)
            narrative_source = "openai"
        except Exception as exc:
            if require_openai:
                raise
            fallback_reason = f"openai_call_failed: {type(exc).__name__}: {exc}"
            narrative = fallback_scenario_narrative(report_context)
            narrative_source = "fallback"
    else:
        if not use_openai:
            fallback_reason = "use_openai_false"
        elif not openai_api_key_present:
            fallback_reason = "missing_openai_api_key"
        narrative = fallback_scenario_narrative(report_context)
        narrative_source = "fallback"

    report_markdown = render_markdown_report(report_context, narrative)

    return {
        "report_context": report_context,
        "narrative": narrative,
        "narrative_source": narrative_source,
        "fallback_reason": fallback_reason,
        "openai_requested": bool(use_openai),
        "openai_api_key_present": openai_api_key_present,
        "openai_model": model,
        "report_markdown": report_markdown,
    }
