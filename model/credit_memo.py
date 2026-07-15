from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CREDIT_MEMO_MODEL = os.getenv("OPENAI_CREDIT_MEMO_MODEL", "gpt-4o-mini")
DEFAULT_RATING_CONTEXT_PATH = os.getenv(
    "OBLIGOR_RATING_CONTEXT_PATH",
    "market_data/obligor_rating_context.parquet",
)


CREDIT_MEMO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "borrower_name": {"type": "string"},
        "memo_label": {"type": "string"},
        "executive_summary": {"type": "string"},
        "request_summary": {"type": "string"},
        "business_profile": {"type": "string"},
        "capital_benchmark_rating_assessment": {"type": "string"},
        "rating_driver_commentary": {"type": "string"},
        "positive_rating_drivers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "negative_rating_drivers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "neutral_rating_diagnostics": {
            "type": "array",
            "items": {"type": "string"},
        },
        "financial_watchpoints": {
            "type": "array",
            "items": {"type": "string"},
        },
        "financial_risk_assessment": {"type": "string"},
        "peer_and_anchor_context": {"type": "string"},
        "key_credit_strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "key_credit_watchpoints": {
            "type": "array",
            "items": {"type": "string"},
        },
        "questions_for_relationship_manager": {
            "type": "array",
            "items": {"type": "string"},
        },
        "credit_committee_focus_areas": {
            "type": "array",
            "items": {"type": "string"},
        },
        "data_quality_and_limitations": {"type": "string"},
    },
    "required": [
        "title",
        "borrower_name",
        "memo_label",
        "executive_summary",
        "request_summary",
        "business_profile",
        "capital_benchmark_rating_assessment",
        "rating_driver_commentary",
        "financial_risk_assessment",
        "peer_and_anchor_context",
        "key_credit_strengths",
        "key_credit_watchpoints",
        "questions_for_relationship_manager",
        "credit_committee_focus_areas",
        "data_quality_and_limitations",
        "positive_rating_drivers",
        "negative_rating_drivers",
        "neutral_rating_diagnostics",
        "financial_watchpoints",
    ],
    "additionalProperties": False,
}


CREDIT_MEMO_INSTRUCTIONS = """
You are a senior corporate credit officer preparing a controlled first-draft credit memo.

Use only the supplied JSON context. Do not invent facts, financial metrics, ratings,
facility terms, news items, or recommendations. Do not use outside knowledge of the
borrower. If a data item is missing, say it is missing and identify it as a follow-up
question or limitation.

Use only the supplied memo_context. Do not infer facts from the borrower name, industry, ticker, or general knowledge.
Treat the credit request as illustrative unless relationship_context or request_status explicitly says it is an actual live request.
Do not say financial performance is volatile unless supplied financial volatility or earnings history supports that. Equity volatility is a market-implied risk metric, not the same as financial performance volatility.
Use the business_summary only to describe business activities, geographies, products and segments.
Unless credit_request.relationship_context explicitly states this is a live borrower request, describe the facility as illustrative or assumed. Do not say the borrower is seeking, requesting, or approaching the bank.
Do not describe trends, deterioration, improvement, operational inefficiency, market competitiveness, market leadership, market size, resilience or strategic importance unless explicitly supported by supplied context.
Distinguish rating model drivers from broader financial watchpoints. If a rating_driver_sensitivity is neutral, do not describe it as a key rating driver.

For the executive summary, always include:
1. CB rating and CB PD.
2. Whether agency rating is available.
3. Main positive rating drivers.
4. Main financial watchpoints.
5. Exposure-level expected loss if supplied.

Use “Capital Benchmark proprietary rating estimate” rather than “internal capital benchmark rating” or “agency-equivalent rating”.

Distinguish clearly between rating model drivers and broader financial watchpoints.
Rating model drivers must come from memo_context.rating_driver_groups. Do not describe a financial metric as a rating driver unless it appears in the positive or negative rating driver groups.
If a driver appears in memo_context.rating_driver_groups.neutral, you may mention it as a neutral diagnostic, but do not call it a key rating driver.
Financial watchpoints must come from memo_context.financial_watchpoints or explicitly supplied credit metrics. Do not invent trends, causes, operational inefficiencies, market competitiveness, or deterioration.

In the output:
- positive_rating_drivers must summarize memo_context.rating_driver_groups.positive.
- negative_rating_drivers must summarize memo_context.rating_driver_groups.negative.
- neutral_rating_diagnostics must summarize memo_context.rating_driver_groups.neutral only where useful.
- financial_watchpoints must summarize memo_context.financial_watchpoints.
- key_credit_watchpoints should be a concise combined credit-officer summary, not a duplicate of every metric.

Do not approve or decline the credit request. Instead, provide:
- preliminary credit assessment,
- rating assessment,
- financial risk assessment,
- key strengths,
- key watchpoints,
- questions for the relationship manager,
- credit committee focus areas.

The backend owns all numbers, tables, and calculations. Your job is to narrate the
supplied context into the JSON fields required by the schema. Use concise, institutional
bank-credit language. Avoid hype, investment advice, trading views, or false precision.

Style rules:
- Use compact figures such as "$100.0mn", "$8.0bn", "0.15%", and "8.8x".
- Mention data-quality flags where relevant.
- Where driver sensitivities are supplied, frame them as model diagnostics, not causal proof.
- If facility terms are missing, make the memo clearly preliminary.
""".strip()


# -----------------------------
# Generic helpers
# -----------------------------


def _build_rating_driver_groups(
    drivers: list[dict[str, Any]],
    materiality_threshold: float = 0.25,
) -> dict[str, list[dict[str, Any]]]:
    """
    Split local rating diagnostics into positive, negative and neutral buckets.

    These are model diagnostics, not causal proof. The point is to stop the LLM
    treating every financial issue as a model driver.
    """
    positive = []
    negative = []
    neutral = []

    for driver in drivers:
        label = _clean_text(driver.get("label"))
        effect = _safe_float(driver.get("rating_effect_vs_median"))
        interpretation = _clean_text(driver.get("interpretation"))

        if not label or effect is None:
            continue

        item = {
            "label": label,
            "effect_notches": round(effect, 2),
            "interpretation": interpretation,
            "direction": _clean_text(driver.get("direction")),
        }

        if effect >= materiality_threshold and interpretation == "supportive":
            positive.append(item)
        elif effect <= -materiality_threshold and interpretation == "negative":
            negative.append(item)
        else:
            neutral.append(item)

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
    }


def _format_metric_value(value: Any, fmt: str | None) -> str:
    if fmt == "percent":
        return format_percent(value)
    if fmt == "multiple":
        return format_multiple(value)
    return str(value) if value is not None else "n/a"


def _build_financial_watchpoints(
    metric_assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build broader financial watchpoints from credit metrics.

    These are not necessarily CB rating model drivers. They are credit-officer
    review points for the memo.
    """
    watchpoints = []

    for metric in metric_assessments:
        field = metric.get("field")
        name = metric.get("metric")
        value = _safe_float(metric.get("value"))
        peer_value = _safe_float(metric.get("peer_value"))
        fmt = metric.get("format")

        if value is None:
            continue

        value_text = _format_metric_value(value, fmt)
        peer_text = _format_metric_value(peer_value, fmt) if peer_value is not None else None

        if field == "debt_to_revenue":
            if peer_value is not None and value > peer_value * 1.25:
                watchpoints.append({
                    "metric": name,
                    "severity": "medium",
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} is above the industry median.",
                })
            elif value >= 1.5:
                watchpoints.append({
                    "metric": name,
                    "severity": "medium",
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} is elevated and should be reviewed.",
                })

        elif field == "debt_to_ebitda":
            if value >= 6.0:
                severity = "high"
            elif value >= 4.0:
                severity = "medium"
            else:
                severity = None

            if severity:
                watchpoints.append({
                    "metric": name,
                    "severity": severity,
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} is elevated relative to normal corporate credit thresholds.",
                })

        elif field == "net_debt_to_ebitda":
            if value >= 6.0:
                severity = "high"
            elif value >= 4.0:
                severity = "medium"
            else:
                severity = None

            if severity:
                watchpoints.append({
                    "metric": name,
                    "severity": severity,
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} indicates elevated net leverage.",
                })

        elif field == "cash_to_debt":
            if value < 0.10:
                watchpoints.append({
                    "metric": name,
                    "severity": "medium",
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} is low, indicating limited cash coverage of total debt.",
                })

        elif field == "profitMargins":
            if value < 0:
                watchpoints.append({
                    "metric": name,
                    "severity": "medium",
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} is negative in the supplied data.",
                })
            elif peer_value is not None and value < peer_value * 0.5:
                watchpoints.append({
                    "metric": name,
                    "severity": "medium",
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} is materially below the industry median.",
                })

        elif field == "annual_vol_5y":
            if peer_value is not None and value > peer_value * 1.25:
                watchpoints.append({
                    "metric": name,
                    "severity": "medium",
                    "value": value_text,
                    "peer_value": peer_text,
                    "comment": f"{name} is above the industry median, indicating elevated market-implied risk.",
                })

    return watchpoints

def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if value == "" or value.lower() in {"nan", "none", "null"}:
                return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
    return default


def _clean_text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        # numpy scalar support without importing numpy
        if hasattr(value, "item"):
            return _json_safe(value.item())
    except Exception:
        pass
    return value


def _first_present(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row:
            value = row.get(name)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return default


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            return []
    return []


def _truncate(text: Any, max_chars: int = 1500) -> str | None:
    text = _clean_text(text)
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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


# -----------------------------
# Data loading and row lookup
# -----------------------------


@lru_cache(maxsize=4)
def load_rating_context(path: str = DEFAULT_RATING_CONTEXT_PATH) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        # Convenience fallbacks for local/prod layouts.
        candidates = [
            Path("data/obligor_rating_context.parquet"),
            Path("market_data/obligor_rating_context.csv"),
            Path("data/obligor_rating_context.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                file_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Could not find obligor rating context file. Tried: {path}, "
                "data/obligor_rating_context.parquet, market_data/obligor_rating_context.csv, "
                "and data/obligor_rating_context.csv."
            )

    if file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    if "symbol" not in df.columns:
        raise ValueError("obligor_rating_context must include a 'symbol' column")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df


def get_obligor_rating_row(symbol: str, path: str = DEFAULT_RATING_CONTEXT_PATH) -> dict[str, Any]:
    if not symbol:
        raise ValueError("symbol is required")
    df = load_rating_context(path)
    clean_symbol = str(symbol).upper().strip()
    matches = df[df["symbol"] == clean_symbol]
    if matches.empty:
        raise LookupError(f"Symbol not found in obligor rating context: {clean_symbol}")
    return matches.iloc[0].to_dict()


# -----------------------------
# Context builder
# -----------------------------


def _build_metric_assessments(row: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = [
        {
            "metric": "Debt / revenue",
            "field": "debt_to_revenue",
            "value": _safe_float(row.get("debt_to_revenue")),
            "format": "multiple",
            "direction": "higher leverage is generally negative",
            "peer_value": _safe_float(row.get("industry_median_debt_to_revenue")),
            "peer_label": "industry median debt / revenue",
        },
        {
            "metric": "Debt / EBITDA",
            "field": "debt_to_ebitda",
            "value": _safe_float(row.get("debt_to_ebitda")),
            "format": "multiple",
            "direction": "higher leverage is generally negative",
            "peer_value": None,
            "peer_label": None,
        },
        {
            "metric": "Net debt / EBITDA",
            "field": "net_debt_to_ebitda",
            "value": _safe_float(row.get("net_debt_to_ebitda")),
            "format": "multiple",
            "direction": "higher leverage is generally negative",
            "peer_value": None,
            "peer_label": None,
        },
        {
            "metric": "Cash / debt",
            "field": "cash_to_debt",
            "value": _safe_float(row.get("cash_to_debt")),
            "format": "percent",
            "direction": "higher liquidity is generally supportive",
            "peer_value": None,
            "peer_label": None,
        },
        {
            "metric": "Profit margin",
            "field": "profitMargins",
            "value": _safe_float(row.get("profitMargins")),
            "format": "percent",
            "direction": "higher profitability is generally supportive",
            "peer_value": _safe_float(row.get("industry_median_profitMargins")),
            "peer_label": "industry median profit margin",
        },
        {
            "metric": "Five-year annual volatility",
            "field": "annual_vol_5y",
            "value": _safe_float(row.get("annual_vol_5y")),
            "format": "percent",
            "direction": "higher market-implied risk is generally negative",
            "peer_value": _safe_float(row.get("industry_median_annual_vol_5y")),
            "peer_label": "industry median annual volatility",
        },
    ]
    return [m for m in metrics if m["value"] is not None]


def _build_rating_feature_context(row: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        ("log_revenue", "Scale / revenue size", "higher is generally supportive"),
        ("profitMargins", "Profitability", "higher is generally supportive"),
        ("annual_vol_5y", "Equity volatility / market-implied risk", "higher is generally negative"),
        ("relative_debt", "Leverage relative to industry", "higher is generally negative"),
        ("industry_rating_feature", "Industry credit-quality anchor", "higher is generally supportive"),
        ("country_rating_feature", "Country credit-quality anchor", "higher is generally supportive"),
    ]
    output = []
    for field, label, direction in fields:
        value = _safe_float(row.get(field))
        if value is None:
            continue
        item: dict[str, Any] = {
            "field": field,
            "label": label,
            "value": value,
            "direction": direction,
        }
        if field == "industry_rating_feature":
            item["rating_label"] = _clean_text(row.get("industry_rating_feature_label"))
        if field == "country_rating_feature":
            item["rating_label"] = _clean_text(row.get("country_rating_feature_label"))
        output.append(item)
    return output


def _clean_driver_sensitivities(row: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    raw = _parse_json_list(row.get("rating_driver_sensitivities_json"))
    cleaned = []
    for item in raw:
        cleaned.append(
            {
                "feature": _clean_text(item.get("feature")),
                "label": _clean_text(item.get("label")),
                "direction": _clean_text(item.get("direction")),
                "current_value": _safe_float(item.get("current_value")),
                "median_value": _safe_float(item.get("median_value")),
                "rating_effect_vs_median": _safe_float(item.get("rating_effect_vs_median")),
                "interpretation": _clean_text(item.get("interpretation")),
            }
        )
    cleaned = [x for x in cleaned if x.get("label")]
    return cleaned[:limit]


def _proposed_exposure(request: dict[str, Any]) -> float | None:
    proposed = _safe_float(request.get("proposed_exposure_usd"))
    if proposed is not None:
        return proposed
    existing = _safe_float(request.get("existing_exposure_usd"), 0.0) or 0.0
    increase = _safe_float(request.get("requested_increase_usd"), 0.0) or 0.0
    if existing > 0 or increase > 0:
        return existing + increase
    return None


def build_credit_memo_context(
    symbol: str,
    credit_request: dict[str, Any] | None = None,
    rating_context_path: str = DEFAULT_RATING_CONTEXT_PATH,
    lgd: float = 0.45,
) -> dict[str, Any]:
    row = get_obligor_rating_row(symbol, rating_context_path)
    credit_request = dict(credit_request or {})

    cb_pd = _safe_float(_first_present(row, "CB pd", "cb_pd", "CB_pd"))
    agency_pd = _safe_float(_first_present(row, "Agency pd", "agency_pd"))
    proposed_exposure = _proposed_exposure(credit_request)
    requested_increase = _safe_float(credit_request.get("requested_increase_usd"))
    lgd = _safe_float(credit_request.get("lgd"), lgd) or 0.45

    exposure_analytics = {
        "existing_exposure_usd": _safe_float(credit_request.get("existing_exposure_usd")),
        "requested_increase_usd": requested_increase,
        "proposed_exposure_usd": proposed_exposure,
        "lgd": lgd,
        "base_pd": cb_pd,
        "base_expected_loss_on_proposed_exposure": (
            proposed_exposure * cb_pd * lgd
            if proposed_exposure is not None and cb_pd is not None
            else None
        ),
        "base_expected_loss_on_requested_increase": (
            requested_increase * cb_pd * lgd
            if requested_increase is not None and cb_pd is not None
            else None
        ),
    }

    missing_features = row.get("missing_rating_features")
    if isinstance(missing_features, str):
        try:
            missing_features = json.loads(missing_features.replace("'", '"'))
        except Exception:
            missing_features = [missing_features] if missing_features else []
    if not isinstance(missing_features, list):
        missing_features = []

    metric_assessments = _build_metric_assessments(row)
    rating_driver_sensitivities = _clean_driver_sensitivities(row)
    rating_driver_groups = _build_rating_driver_groups(rating_driver_sensitivities)
    financial_watchpoints = _build_financial_watchpoints(metric_assessments)

    context = {
        "memo_version": "credit_memo_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "borrower": {
            "symbol": _clean_text(row.get("symbol")),
            "company_name": _clean_text(row.get("company_name")),
            "business_summary": _truncate(row.get("longBusinessSummary"), 1500),
            "industry": _clean_text(row.get("industry")),
            "sector": _clean_text(row.get("sector")),
            "country": _clean_text(row.get("country")),
            "is_rated_obligor": _safe_bool(row.get("is_rated_obligor")),
        },
        "credit_request": {
            "request_type": _clean_text(credit_request.get("request_type"), "preliminary_credit_assessment"),
            "existing_exposure_usd": _safe_float(credit_request.get("existing_exposure_usd")),
            "requested_increase_usd": requested_increase,
            "proposed_exposure_usd": proposed_exposure,
            "facility_type": _clean_text(credit_request.get("facility_type")),
            "purpose": _clean_text(credit_request.get("purpose")),
            "tenor_years": _safe_float(credit_request.get("tenor_years")),
            "secured": _safe_bool(credit_request.get("secured")),
            "seniority": _clean_text(credit_request.get("seniority")),
            "relationship_context": _clean_text(credit_request.get("relationship_context")),
            "currency": _clean_text(credit_request.get("currency"), "USD"),
        },
        "capital_benchmark_rating": {
            "model_version": _clean_text(row.get("rating_model_version")),
            "model_type": _clean_text(row.get("rating_model_type")),
            "cb_rating": _clean_text(row.get("cb_rating")),
            "cb_pd": cb_pd,
            "cb_pd_percent": cb_pd * 100 if cb_pd is not None else None,
            "cb_rating_num_raw": _safe_float(row.get("cb_rating_num_raw")),
            "cb_rating_num": _safe_float(row.get("cb_rating_num")),
            "cb_rating_notch_margin": _safe_float(row.get("cb_rating_notch_margin")),
            "cb_rating_boundary_comment": _clean_text(row.get("cb_rating_boundary_comment")),
            "agency_rating": _clean_text(_first_present(row, "Agency Rating", "agency_rating")),
            "agency_pd": agency_pd,
            "cb_minus_agency_notches": _safe_float(row.get("cb_minus_agency_notches")),
            "cb_vs_agency_comment": _clean_text(row.get("cb_vs_agency_comment")),
            "rating_context_quality": _clean_text(row.get("rating_context_quality")),
            "scored_with_imputation": _safe_bool(row.get("scored_with_imputation")),
        },
        "financials": {
            "total_debt_usd": _safe_float(row.get("totalDebt_usd")),
            "total_revenue_usd": _safe_float(row.get("totalRevenue_usd")),
            "total_cash_usd": _safe_float(row.get("totalCash_usd")),
            "net_debt_usd": _safe_float(row.get("net_debt_usd")),
            "ebitda_usd": _safe_float(row.get("ebitda_usd")),
            "market_cap_usd": _safe_float(row.get("marketCap_usd")),
            "enterprise_value_usd": _safe_float(row.get("enterpriseValue_usd")),
            "net_income_to_common_usd": _safe_float(row.get("netIncomeToCommon_usd")),
        },
        "credit_metrics": {
            "profit_margin": _safe_float(row.get("profitMargins")),
            "debt_to_revenue": _safe_float(row.get("debt_to_revenue")),
            "debt_to_ebitda": _safe_float(row.get("debt_to_ebitda")),
            "cash_to_debt": _safe_float(row.get("cash_to_debt")),
            "net_debt_to_ebitda": _safe_float(row.get("net_debt_to_ebitda")),
            "relative_debt": _safe_float(row.get("relative_debt")),
            "annual_vol_5y": _safe_float(row.get("annual_vol_5y")),
        },
        "model_features": _build_rating_feature_context(row),
        "metric_assessments": metric_assessments,
        "rating_driver_sensitivities": rating_driver_sensitivities,
        "rating_driver_groups": rating_driver_groups,
        "financial_watchpoints": financial_watchpoints,
        "peer_and_anchor_context": {
            "industry_rating_anchor": _clean_text(row.get("industry_rating_feature_label")),
            "country_rating_anchor": _clean_text(row.get("country_rating_feature_label")),
            "industry_median_debt_to_revenue": _safe_float(row.get("industry_median_debt_to_revenue")),
            "industry_median_annual_vol_5y": _safe_float(row.get("industry_median_annual_vol_5y")),
            "industry_median_profit_margin": _safe_float(row.get("industry_median_profitMargins")),
            "industry_median_revenue_usd": _safe_float(row.get("industry_median_totalRevenue_usd")),
            "revenue_percentile_global": _safe_float(row.get("revenue_percentile_global")),
            "profit_margin_percentile_global": _safe_float(row.get("profit_margin_percentile_global")),
            "volatility_percentile_global": _safe_float(row.get("volatility_percentile_global")),
            "relative_debt_percentile_global": _safe_float(row.get("relative_debt_percentile_global")),
            "revenue_percentile_industry": _safe_float(row.get("revenue_percentile_industry")),
            "profit_margin_percentile_industry": _safe_float(row.get("profit_margin_percentile_industry")),
            "volatility_percentile_industry": _safe_float(row.get("volatility_percentile_industry")),
            "relative_debt_percentile_industry": _safe_float(row.get("relative_debt_percentile_industry")),
        },
        "exposure_analytics": exposure_analytics,
        "data_quality": {
            "n_missing_rating_features": _safe_float(row.get("n_missing_rating_features")),
            "missing_rating_features": missing_features,
            "scored_with_imputation": _safe_bool(row.get("scored_with_imputation")),
            "rating_context_quality": _clean_text(row.get("rating_context_quality")),
            "warnings": [],
        },
    }

    if not context["capital_benchmark_rating"].get("agency_rating"):
        context["data_quality"]["warnings"].append(
            "No agency rating is available in the supplied rating context."
        )
    if proposed_exposure is None:
        context["data_quality"]["warnings"].append(
            "No proposed exposure is available; exposure-level expected loss is not calculated."
        )

    return _json_safe(context)


# -----------------------------
# OpenAI call and fallback
# -----------------------------


def generate_credit_memo_narrative(
    memo_context: dict[str, Any],
    model: str = DEFAULT_CREDIT_MEMO_MODEL,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=CREDIT_MEMO_INSTRUCTIONS,
        input=json.dumps(memo_context, separators=(",", ":"), ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "credit_memo_narrative",
                "strict": True,
                "schema": CREDIT_MEMO_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


def fallback_credit_memo_narrative(memo_context: dict[str, Any]) -> dict[str, Any]:
    borrower = memo_context.get("borrower", {})
    rating = memo_context.get("capital_benchmark_rating", {})
    metrics = memo_context.get("credit_metrics", {})
    request = memo_context.get("credit_request", {})
    drivers = memo_context.get("rating_driver_sensitivities", [])
    driver_groups = memo_context.get("rating_driver_groups", {})
    financial_watchpoints_context = memo_context.get("financial_watchpoints", [])

    positive_rating_drivers = [
        f"{d.get('label')} is supportive in the local rating diagnostics "
        f"({d.get('effect_notches'):+.2f} notches vs median)."
        for d in driver_groups.get("positive", [])
    ]

    negative_rating_drivers = [
        f"{d.get('label')} is negative in the local rating diagnostics "
        f"({d.get('effect_notches'):+.2f} notches vs median)."
        for d in driver_groups.get("negative", [])
    ]

    neutral_rating_diagnostics = [
        f"{d.get('label')} is broadly neutral in the local rating diagnostics "
        f"({d.get('effect_notches'):+.2f} notches vs median)."
        for d in driver_groups.get("neutral", [])[:4]
    ]

    financial_watchpoints = [
        f"{w.get('metric')}: {w.get('value')} — {w.get('comment')}"
        for w in financial_watchpoints_context
    ]

    borrower_name = borrower.get("company_name") or borrower.get("symbol") or "Borrower"
    cb_rating = rating.get("cb_rating") or "n/a"
    cb_pd = rating.get("cb_pd")
    facility_type = request.get("facility_type") or "facility"

    supportive = [d for d in drivers if d.get("interpretation") == "supportive"][:3]
    negative = [d for d in drivers if d.get("interpretation") == "negative"][:3]

    strengths = [
        f"Capital Benchmark rating estimate of {cb_rating}.",
        "Rating context data is available for the borrower.",
    ]
    strengths.extend([f"{d.get('label')} is supportive in the local rating diagnostics." for d in supportive])

    watchpoints = []
    watchpoints.extend([f"{d.get('label')} is a negative factor in the local rating diagnostics." for d in negative])
    if metrics.get("debt_to_ebitda") is not None:
        watchpoints.append(f"Debt / EBITDA is {format_multiple(metrics.get('debt_to_ebitda'))} and should be reviewed.")
    if not request.get("proposed_exposure_usd"):
        watchpoints.append("Facility size and proposed exposure are not supplied.")

    return {
        "title": "Credit Memo",
        "borrower_name": str(borrower_name),
        "memo_label": "Preliminary Credit Assessment",
        "executive_summary": (
            f"{borrower_name} has a Capital Benchmark rating estimate of {cb_rating}"
            + (f", corresponding to a CB PD of {format_percent(cb_pd)}" if cb_pd is not None else "")
            + ". The memo is preliminary and based on supplied rating-context data."
        ),
        "request_summary": (
            f"The request is recorded as {facility_type}. Key facility terms should be supplied "
            "before the memo is used for a credit decision."
        ),
        "positive_rating_drivers": positive_rating_drivers or [
            "No material positive rating drivers were identified from the supplied local diagnostics."
        ],
        "negative_rating_drivers": negative_rating_drivers or [
            "No material negative rating drivers were identified from the supplied local diagnostics."
        ],
        "neutral_rating_diagnostics": neutral_rating_diagnostics,
        "financial_watchpoints": financial_watchpoints or [
            "No deterministic financial watchpoints were generated from the supplied metrics."
        ],
        "business_profile": (
            f"{borrower_name} operates in {borrower.get('industry') or 'the supplied industry'} "
            f"within {borrower.get('sector') or 'the supplied sector'}."
        ),
        "capital_benchmark_rating_assessment": (
            f"The Capital Benchmark rating estimate is {cb_rating}. This is a proprietary "
            "internal rating estimate and should not be described as an agency rating."
        ),
        "rating_driver_commentary": (
            "The main rating diagnostics should be reviewed using the supplied driver sensitivities. "
            "These sensitivities are model diagnostics rather than causal attribution."
        ),
        "financial_risk_assessment": (
            "The supplied financial metrics indicate the key balance-sheet and profitability inputs "
            "behind the rating estimate. Leverage, liquidity and profitability should be checked "
            "against recent financial statements."
        ),
        "peer_and_anchor_context": (
            "The peer and anchor context provides industry and country rating anchors plus "
            "industry-median metrics for comparison."
        ),
        "key_credit_strengths": strengths[:6],
        "key_credit_watchpoints": (
            financial_watchpoints[:4]
            or negative_rating_drivers[:4]
            or ["No specific watchpoints were generated from the supplied fallback context."]
        ),
        "questions_for_relationship_manager": [
            "What is the requested facility size, tenor, purpose, seniority and security package?",
            "What is the borrower's current debt maturity profile and covenant position?",
            "Are recent financial metrics recurring or affected by one-off items?",
            "Are there recent adverse news, litigation, regulatory or rating-action issues?",
        ],
        "credit_committee_focus_areas": [
            "Validate the CB rating estimate against the bank's internal rating view.",
            "Assess leverage, profitability and liquidity metrics in the latest financial statements.",
            "Review any proposed exposure increase against limits, risk appetite and concentration.",
        ],
        "data_quality_and_limitations": (
            "This memo is based only on supplied Capital Benchmark rating-context data. It does not "
            "include full financial statements, covenant analysis, collateral review, legal review, "
            "recent news screening or a final credit recommendation."
        ),
    }


def create_credit_memo(
    symbol: str,
    credit_request: dict[str, Any] | None = None,
    use_openai: bool = True,
    require_openai: bool = False,
    model: str | None = None,
    rating_context_path: str = DEFAULT_RATING_CONTEXT_PATH,
) -> dict[str, Any]:
    memo_context = build_credit_memo_context(
        symbol=symbol,
        credit_request=credit_request,
        rating_context_path=rating_context_path,
    )

    selected_model = model or DEFAULT_CREDIT_MEMO_MODEL
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    fallback_reason = None
    narrative_source = "fallback"

    if use_openai and api_key_present:
        try:
            narrative = generate_credit_memo_narrative(memo_context, model=selected_model)
            narrative_source = "openai"
        except Exception as exc:
            fallback_reason = f"openai_error: {type(exc).__name__}: {exc}"
            if require_openai:
                raise
            narrative = fallback_credit_memo_narrative(memo_context)
    else:
        if not use_openai:
            fallback_reason = "openai_disabled"
        elif not api_key_present:
            fallback_reason = "missing_openai_api_key"
        if require_openai:
            raise RuntimeError(f"OpenAI narrative required but unavailable: {fallback_reason}")
        narrative = fallback_credit_memo_narrative(memo_context)

    markdown = render_credit_memo_markdown(memo_context, narrative)

    return _json_safe(
        {
            "narrative_source": narrative_source,
            "fallback_reason": fallback_reason,
            "openai_requested": bool(use_openai),
            "openai_api_key_present": api_key_present,
            "openai_model": selected_model,
            "memo_context": memo_context,
            "narrative": narrative,
            "memo_markdown": markdown,
        }
    )


# -----------------------------
# Markdown renderer
# -----------------------------


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _none_to_na(value: Any) -> str:
    text = _clean_text(value)
    return text if text is not None else "n/a"


def _request_table(context: dict[str, Any]) -> str:
    request = context.get("credit_request", {})
    exposure = context.get("exposure_analytics", {})
    rows = [
        ["Request type", _none_to_na(request.get("request_type"))],
        ["Facility type", _none_to_na(request.get("facility_type"))],
        ["Purpose", _none_to_na(request.get("purpose"))],
        ["Existing exposure", format_currency(request.get("existing_exposure_usd"))],
        ["Requested increase", format_currency(request.get("requested_increase_usd"))],
        ["Proposed exposure", format_currency(request.get("proposed_exposure_usd"))],
        ["Tenor", f"{request.get('tenor_years')} years" if request.get("tenor_years") is not None else "n/a"],
        ["Secured", _none_to_na(request.get("secured"))],
        ["Base EL on proposed exposure", format_currency(exposure.get("base_expected_loss_on_proposed_exposure"))],
    ]
    return _markdown_table(["Field", "Value"], rows)


def _rating_table(context: dict[str, Any]) -> str:
    rating = context.get("capital_benchmark_rating", {})
    rows = [
        ["CB rating", _none_to_na(rating.get("cb_rating"))],
        ["CB PD", format_percent(rating.get("cb_pd"))],
        ["Raw model score", f"{rating.get('cb_rating_num_raw'):.2f}" if rating.get("cb_rating_num_raw") is not None else "n/a"],
        ["Agency rating", _none_to_na(rating.get("agency_rating"))],
        ["CB vs agency", _none_to_na(rating.get("cb_vs_agency_comment"))],
        ["Rating context quality", _none_to_na(rating.get("rating_context_quality"))],
        ["Model", _none_to_na(rating.get("model_type"))],
        ["Model version", _none_to_na(rating.get("model_version"))],
    ]
    return _markdown_table(["Metric", "Value"], rows)


def _financial_table(context: dict[str, Any]) -> str:
    financials = context.get("financials", {})
    metrics = context.get("credit_metrics", {})
    rows = [
        ["Revenue", format_currency(financials.get("total_revenue_usd"))],
        ["Total debt", format_currency(financials.get("total_debt_usd"))],
        ["Cash", format_currency(financials.get("total_cash_usd"))],
        ["Net debt", format_currency(financials.get("net_debt_usd"))],
        ["EBITDA", format_currency(financials.get("ebitda_usd"))],
        ["Market cap", format_currency(financials.get("market_cap_usd"))],
        ["Debt / revenue", format_multiple(metrics.get("debt_to_revenue"))],
        ["Debt / EBITDA", format_multiple(metrics.get("debt_to_ebitda"))],
        ["Net debt / EBITDA", format_multiple(metrics.get("net_debt_to_ebitda"))],
        ["Cash / debt", format_percent(metrics.get("cash_to_debt"))],
        ["Profit margin", format_percent(metrics.get("profit_margin"))],
        ["Annual volatility", format_percent(metrics.get("annual_vol_5y"))],
    ]
    return _markdown_table(["Metric", "Value"], rows)


def _driver_table(context: dict[str, Any]) -> str:
    rows = []
    for item in context.get("rating_driver_sensitivities", [])[:8]:
        effect = item.get("rating_effect_vs_median")
        rows.append(
            [
                _none_to_na(item.get("label")),
                f"{effect:+.2f} notches" if effect is not None else "n/a",
                _none_to_na(item.get("interpretation")),
            ]
        )
    if not rows:
        rows = [["No driver sensitivity data supplied", "n/a", "n/a"]]
    return _markdown_table(["Driver", "Effect vs median", "Assessment"], rows)


def _bullets(items: list[Any]) -> str:
    if not items:
        return "- n/a"
    return "\n".join(f"- {_none_to_na(item)}" for item in items)


def render_credit_memo_markdown(
    memo_context: dict[str, Any],
    narrative: dict[str, Any],
) -> str:
    borrower = memo_context.get("borrower", {})
    return f"""# {narrative.get("title", "Credit Memo")}

**Borrower:** {narrative.get("borrower_name") or borrower.get("company_name") or borrower.get("symbol") or "n/a"}  
**Memo type:** {narrative.get("memo_label", "Preliminary Credit Assessment")}

## Executive Summary

{narrative.get("executive_summary", "")}

## Borrower and Request Summary

{narrative.get("request_summary", "")}

{_request_table(memo_context)}

## Business Profile

{narrative.get("business_profile", "")}

## Capital Benchmark Rating Assessment

{narrative.get("capital_benchmark_rating_assessment", "")}

{_rating_table(memo_context)}

## Rating Driver Commentary

{narrative.get("rating_driver_commentary", "")}

{_driver_table(memo_context)}

## Positive Rating Drivers

{_bullets(narrative.get("positive_rating_drivers", []))}

## Negative Rating Drivers

{_bullets(narrative.get("negative_rating_drivers", []))}

## Neutral Rating Diagnostics

{_bullets(narrative.get("neutral_rating_diagnostics", []))}

## Financial Risk Assessment

{narrative.get("financial_risk_assessment", "")}

{_financial_table(memo_context)}

## Financial Watchpoints

{_bullets(narrative.get("financial_watchpoints", []))}

## Peer and Anchor Context

{narrative.get("peer_and_anchor_context", "")}

## Key Credit Strengths

{_bullets(narrative.get("key_credit_strengths", []))}

## Key Credit Watchpoints

{_bullets(narrative.get("key_credit_watchpoints", []))}

## Questions for Relationship Manager

{_bullets(narrative.get("questions_for_relationship_manager", []))}

## Credit Committee Focus Areas

{_bullets(narrative.get("credit_committee_focus_areas", []))}

## Data Quality and Limitations

{narrative.get("data_quality_and_limitations", "")}
""".strip()
