import csv
import json
import os

from bs4 import BeautifulSoup

from flask import Flask, request, jsonify, abort, send_file, url_for
import io
from flask_cors import CORS
from model.stress_model import run_stress
from model.scenario_report import create_scenario_report
from model.scenario_report_documents import (
    DOCX_MIMETYPE,
    PDF_MIMETYPE,
    convert_docx_to_pdf,
    render_scenario_report_docx,
    scenario_report_filename,
)
from model.credit_memo import create_credit_memo, load_rating_context, load_credit_policy
from model.credit_memo_annotation import annotate_credit_memo, annotation_config
from model.credit_memo_documents import (
    DOCX_MIMETYPE,
    PDF_MIMETYPE,
    convert_docx_to_pdf,
    credit_memo_filename,
    render_credit_memo_docx,
)


from io import BytesIO
from datetime import datetime
import zipfile

from pathlib import Path
import tempfile

import pandas as pd


app = Flask(__name__)

# Prototype: allow Bubble / browser calls
CORS(app)


BENCHMARK_COMPANIES = [
    {"api": "GOOG", "display_name": "Alphabet"},
    {"api": "NVDA", "display_name": "NVIDIA"},
    {"api": "RYDAF", "display_name": "Shell"},
    {"api": "BP", "display_name": "BP"},
    {"api": "VOD.L", "display_name": "Vodafone"},
    {"api": "BT-A.L", "display_name": "BT Group"},
    {"api": "TSCDF", "display_name": "Tesco"},
    {"api": "MAKSY", "display_name": "Marks & Spencer"},
    {"api": "IAG.VI", "display_name": "International Airlines Group"},
    {"api": "RR.L", "display_name": "Rolls-Royce"},
    {"api": "VWAGY", "display_name": "Volkswagen"},
    {"api": "RNLSY", "display_name": "Renault"},
    {"api": "STLA", "display_name": "Stellantis"},
    {"api": "TUIFF", "display_name": "TUI"},
    {"api": "NOK", "display_name": "Nokia"},
    {"api": "BASFY", "display_name": "BASF"},
    {"api": "AF.PA", "display_name": "Air France-KLM"},
    {"api": "EADSY", "display_name": "Airbus"},
    {"api": "MT", "display_name": "ArcelorMittal"},
    {"api": "AAL", "display_name": "American Airlines"},
]

BENCHMARK_MODELS = [
    {"api": "mini", "display_name": "GPT-4o-mini"},
]

BENCHMARK_CONTEXTS = [
    {"api": "full", "display_name": "Full Context"},
    {"api": "minimal", "display_name": "Minimal Context"},
]

BENCHMARK_EVALUATION_MODES = [
    {"api": "none", "display_name": "None"},
    {"api": "llm_evaluated", "display_name": "LLM Evaluated"},
    {
        "api": "deterministic_evaluated",
        "display_name": "Deterministic Evaluated",
    },
]

BENCHMARK_PROMPT_MODES = [
    {"api": "tight", "display_name": "Tight"},
    {"api": "loose", "display_name": "Loose"},
]

BENCHMARK_COMPANY_LABELS = {
    item["api"]: item["display_name"]
    for item in BENCHMARK_COMPANIES
}

BENCHMARK_MODEL_LABELS = {
    item["api"]: item["display_name"]
    for item in BENCHMARK_MODELS
}

BENCHMARK_CONTEXT_LABELS = {
    item["api"]: item["display_name"]
    for item in BENCHMARK_CONTEXTS
}

BENCHMARK_EVALUATION_LABELS = {
    item["api"]: item["display_name"]
    for item in BENCHMARK_EVALUATION_MODES
}

BENCHMARK_PROMPT_LABELS = {
    item["api"]: item["display_name"]
    for item in BENCHMARK_PROMPT_MODES
}

BENCHMARK_DATA_DIR = Path(
    "/opt/capitalbenchmark-data/credit_memo_benchmark_2026_v1"
)

HTML_DIR = BENCHMARK_DATA_DIR / "rendered_files" / "html"

# BENCHMARK_DATA_DIR = Path("/Users/barrie/capitalbenchmark-api/benchmark_runs/benchmark_20_mini_memos")

# HTML_DIR = BENCHMARK_DATA_DIR / "html"



# ---------------------------------------------------------------------------
# Stored credit memo benchmark endpoints
# ---------------------------------------------------------------------------

def resolve_credit_memo_html_file(memo_id: str) -> Path:
    """
    Resolve one stored HTML memo safely.

    Supports:
    - full stem:
      0001__GOOG__ctx_full__policy_full__prompt_standard__tier_frontier__run_01
    - numeric memo ID:
      0001
    """

    clean_memo_id = str(memo_id).strip()

    if not clean_memo_id:
        abort(400, description="Memo ID is required.")

    # Prevent path traversal and arbitrary filenames.
    if "/" in clean_memo_id or "\\" in clean_memo_id or ".." in clean_memo_id:
        abort(400, description="Invalid memo ID.")

    # Exact filename/stem match.
    exact_path = HTML_DIR / f"{clean_memo_id}.html"

    if exact_path.is_file():
        return exact_path

    # Numeric/short memo ID match, e.g. 0001.
    matches = sorted(HTML_DIR.glob(f"{clean_memo_id}__*.html"))

    if len(matches) == 1:
        return matches[0]

    if not matches:
        abort(404, description=f"HTML memo not found: {clean_memo_id}")

    abort(
        409,
        description=f"Multiple HTML files matched memo ID: {clean_memo_id}",
    )

def get_benchmark_dir() -> Path:
    """
    Return the configured benchmark directory.

    Local example:
        /Users/barrie/capitalbenchmark-api/
        benchmark_runs/benchmark_20_mini_memos

    Production example:
        /opt/capitalbenchmark-data/credit_memo_benchmark_2026_v1
    """
    configured_dir = os.getenv("BENCHMARK_CREDIT_MEMO_DIR")

    if not configured_dir:
        raise RuntimeError(
            "BENCHMARK_CREDIT_MEMO_DIR environment variable is not set."
        )

    benchmark_dir = Path(configured_dir).expanduser().resolve()

    if not benchmark_dir.is_dir():
        raise RuntimeError(
            f"Benchmark directory does not exist: {benchmark_dir}"
        )

    return benchmark_dir


def read_benchmark_index() -> list[dict]:
    """
    Read the annotation summary CSV, which acts as the benchmark index.
    """
    benchmark_dir = get_benchmark_dir()
    index_file = benchmark_dir / "offline_annotation_summary.csv"

    if not index_file.is_file():
        raise FileNotFoundError(
            f"Benchmark index was not found: {index_file}"
        )

    with index_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def get_memo_id(row: dict) -> str:
    """
    Derive the stable memo ID from the raw memo filename.

    Example:
        0001__GOOG__ctx_full__policy_none__prompt_tight__
        tier_mini__run_1__d867977c9f14
    """
    memo_file = row.get("memo_file", "")
    return Path(memo_file).stem


def normalise_index_row(row: dict) -> dict:
    """
    Return the fields needed by the Bubble list view.
    """
    memo_id = get_memo_id(row)
    benchmark_dir = get_benchmark_dir()

    pdf_file = benchmark_dir / "rendered_files" / "pdf" / f"{memo_id}.pdf"
    docx_file = benchmark_dir / "rendered_files" / "docx" / f"{memo_id}.docx"
    memo_file = benchmark_dir / "raw_memos" / f"{memo_id}.json"
    annotation_file = (
        benchmark_dir
        / "offline_annotations"
        / f"{memo_id}__annotation.json"
    )

    def integer_value(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def float_value(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    symbol = row.get("symbol")
    context_mode = row.get("context_mode")
    policy_mode = row.get("policy_mode")
    prompt_mode = row.get("prompt_mode")
    model_tier = row.get("model_tier")

    return {
        "memo_id": memo_id,

        # --- Company ---
        "symbol": symbol,
        "company_name": row.get("company_name"),
        "company_display_name": BENCHMARK_COMPANY_LABELS.get(
            symbol,
            row.get("company_name") or symbol,
        ),

        "industry": row.get("industry"),
        "sector": row.get("sector"),
        "country": row.get("country"),

        # --- Benchmark dimensions (raw API values) ---
        "context_mode": context_mode,
        "policy_mode": policy_mode,
        "prompt_mode": prompt_mode,
        "model_tier": model_tier,
        "model": row.get("model"),

        # --- Friendly display values ---
        "context_display_name": BENCHMARK_CONTEXT_LABELS.get(
            context_mode,
            context_mode,
        ),
        "evaluation_display_name": BENCHMARK_EVALUATION_LABELS.get(
            policy_mode,
            policy_mode,
        ),
        "prompt_display_name": BENCHMARK_PROMPT_LABELS.get(
            prompt_mode,
            prompt_mode,
        ),
        "model_display_name": BENCHMARK_MODEL_LABELS.get(
            model_tier,
            model_tier,
        ),

        # --- Metadata ---
        "run_id": integer_value(row.get("run_id")),
        "experiment_id": row.get("experiment_id"),

        # --- Scores ---
        "overall_score": float_value(row.get("overall_score")),
        "policy_detection_score": float_value(
            row.get("policy_detection_score")
        ),
        "missing_information_detection_score": float_value(
            row.get("missing_information_detection_score")
        ),

        # --- Annotation statistics ---
        "annotation_count": integer_value(
            row.get("annotation_count")
        ),
        "critical_or_high_issue_count": integer_value(
            row.get("critical_or_high_issue_count")
        ),
        "factual_error_count": integer_value(
            row.get("factual_error_count")
        ),
        "unsupported_claim_count": integer_value(
            row.get("unsupported_claim_count")
        ),

        # --- File availability ---
        "files": {
            "memo_json_available": memo_file.is_file(),
            "annotation_json_available": annotation_file.is_file(),
            "pdf_available": pdf_file.is_file(),
            "docx_available": docx_file.is_file(),
        },

        # --- URLs ---
        "urls": {
            "detail": url_for(
                "benchmark_credit_memo_detail",
                memo_id=memo_id,
            ),
            "context": url_for(
                "benchmark_credit_memo_context",
                memo_id=memo_id,
            ),
            "pdf": url_for(
                "benchmark_credit_memo_file",
                memo_id=memo_id,
                format="pdf",
            ),
            "docx": url_for(
                "benchmark_credit_memo_file",
                memo_id=memo_id,
                format="docx",
            ),
        },
    }


def find_benchmark_row(memo_id: str) -> dict | None:
    """
    Find one benchmark index row using its memo ID.
    """
    for row in read_benchmark_index():
        if get_memo_id(row) == memo_id:
            return row

    return None


def read_json_file(file_path: Path, description: str):
    """
    Read and parse a stored JSON file.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError: if the file cannot be read or parsed.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} was not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read {description}: {exc}"
        ) from exc


def extract_annotation_items(annotation_payload) -> list:
    """
    Return the annotation findings as a list for Bubble.

    This supports several likely annotation payload shapes while preserving
    the complete raw annotation payload elsewhere in the API response.
    """
    if annotation_payload is None:
        return []

    if isinstance(annotation_payload, list):
        return annotation_payload

    if not isinstance(annotation_payload, dict):
        return []

    for field in (
        "annotations",
        "issues",
        "findings",
        "annotation_items",
    ):
        value = annotation_payload.get(field)

        if isinstance(value, list):
            return value

    nested_annotation = annotation_payload.get("annotation")

    if isinstance(nested_annotation, list):
        return nested_annotation

    if isinstance(nested_annotation, dict):
        for field in (
            "annotations",
            "issues",
            "findings",
            "annotation_items",
        ):
            value = nested_annotation.get(field)

            if isinstance(value, list):
                return value

    return []



def _bool_param(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_param(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _credit_request_from_source(source):
    return {
        "request_type": source.get("request_type", "preliminary_credit_assessment"),
        "existing_exposure_usd": _float_param(source.get("existing_exposure_usd")),
        "requested_increase_usd": _float_param(source.get("requested_increase_usd")),
        "proposed_exposure_usd": _float_param(source.get("proposed_exposure_usd")),
        "facility_type": source.get("facility_type"),
        "purpose": source.get("purpose"),
        "tenor_years": _float_param(source.get("tenor_years")),
        "secured": None if source.get("secured") is None else _bool_param(source.get("secured")),
        "seniority": source.get("seniority"),
        "relationship_context": source.get("relationship_context"),
        "currency": source.get("currency", "USD"),
        "lgd": _float_param(source.get("lgd"), 0.45),
    }

def _truthy(value, default=False):
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "y"}

def _bool_arg(name, default=False):
    value = request.args.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}

def _request_value(payload, name, default=None):
    if isinstance(payload, dict) and name in payload:
        return payload.get(name)
    return request.args.get(name, default)


def _request_float(payload, name, default=0.0):
    value = _request_value(payload, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _request_int(payload, name, default=20):
    value = _request_value(payload, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@app.route("/")
def home():
    return "Capital Benchmark API"


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "capitalbenchmark-api"
    })


@app.route("/presets")
def presets():
    return jsonify({
        "scenarios": [
            {"name": "Current Market", "oil": 1.5, "ai": 3.0},
            {"name": "AI Normalises", "oil": 1.5, "ai": 0.0},
            {"name": "AI Bust", "oil": 1.5, "ai": -1.5},
            {"name": "Oil Shock Only", "oil": 1.5, "ai": 0.0},
            {"name": "Benign", "oil": 0.0, "ai": 0.0}
        ]
    })



@app.route("/stress")
def stress():
    market = float(request.args.get("market", 0))
    technology = float(request.args.get("technology", request.args.get("ai", 0)))
    commodity = float(request.args.get("commodity", request.args.get("oil", 0)))
    lgd = float(request.args.get("lgd", 0.45))
    top_n = int(request.args.get("top_n", 10))

    result = run_stress(
        market=market,
        technology=technology,
        commodity=commodity,
        lgd=lgd,
        top_n=top_n,
    )

    return jsonify(result)

@app.route("/download_scenario")
def download_scenario():
    market = float(request.args.get("market", 0))
    technology = float(request.args.get("technology", request.args.get("ai", 0)))
    commodity = float(request.args.get("commodity", request.args.get("oil", 0)))
    lgd = float(request.args.get("lgd", 0.45))

    # Use a larger default for downloads than for the UI
    top_n = int(request.args.get("top_n", 100))

    result = run_stress(
        market=market,
        technology=technology,
        commodity=commodity,
        lgd=lgd,
        top_n=top_n,
    )

    top_industries = pd.DataFrame(result.get("top_industries", []))[["industry", "obligors", "ead", "base_pd", "stressed_pd", "stressed_expected_loss", "pd_multiple", "rho_Market", "rho_Technology", "rho_Commodity"]]
    top_obligors = pd.DataFrame(result.get("top_obligors", []))[["symbol", "company_name", "Agency Rating", "cb_rating", "industry", "sector", "country", "ead", "base_pd", "stressed_pd", "stressed_expected_loss", "pd_multiple", "rho_Market", "rho_Technology", "rho_Commodity"]]

    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "top_industries.csv",
            top_industries.to_csv(index=False),
        )

        zf.writestr(
            "top_obligors.csv",
            top_obligors.to_csv(index=False),
        )

    buffer.seek(0)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"capital_benchmark_scenario_"
        f"{timestamp}.zip"
    )


    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )

@app.route("/scenario_report", methods=["GET", "POST"])
def scenario_report():
    payload = request.get_json(silent=True) or {}

    market = _request_float(payload, "market", 0.0)
    technology = _request_float(
        payload,
        "technology",
        _request_float(payload, "ai", 0.0),
    )
    commodity = _request_float(
        payload,
        "commodity",
        _request_float(payload, "oil", 0.0),
    )
    lgd = _request_float(payload, "lgd", 0.45)

    report_industries_n = _request_int(payload, "top_industries_n", 20)
    report_obligors_n = _request_int(payload, "top_obligors_n", 20)

    # Use a larger internal top_n so concentration metrics and top tables are stable.
    model_top_n = _request_int(payload, "model_top_n", 250)

    use_openai = str(_request_value(payload, "use_openai", "true")).lower() != "false"
    require_openai = str(_request_value(payload, "require_openai", "false")).lower() == "true"

    stress_result = run_stress(
        market=market,
        technology=technology,
        commodity=commodity,
        lgd=lgd,
        top_n=model_top_n,
    )

    report = create_scenario_report(
        stress_result=stress_result,
        scenario={
            "market": market,
            "technology": technology,
            "commodity": commodity,
            "lgd": lgd,
        },
        use_openai=use_openai,
        require_openai=require_openai,
        top_industries_n=report_industries_n,
        top_obligors_n=report_obligors_n,
    )

    return jsonify(report)

@app.route("/scenario_report_file")
def scenario_report_file():
    file_format = request.args.get("format", "docx").lower().strip()
    if file_format not in {"docx", "pdf"}:
        return jsonify({"error": "format must be 'docx' or 'pdf'"}), 400

    market = float(request.args.get("market", 0))
    technology = float(request.args.get("technology", request.args.get("ai", 0)))
    commodity = float(request.args.get("commodity", request.args.get("oil", 0)))
    lgd = float(request.args.get("lgd", 0.45))

    top_n = int(request.args.get("top_n", 100))
    industry_limit = int(request.args.get("industry_limit", 10))
    obligor_limit = int(request.args.get("obligor_limit", 10))

    use_openai = _bool_arg("use_openai", True)
    require_openai = _bool_arg("require_openai", False)

    stress_result = run_stress(
        market=market,
        technology=technology,
        commodity=commodity,
        lgd=lgd,
        top_n=top_n,
    )

    scenario = {
        "market": market,
        "technology": technology,
        "commodity": commodity,
        "lgd": lgd,
        "top_n": top_n,
    }

    report_payload = create_scenario_report(
        stress_result=stress_result,
        scenario=scenario,
        use_openai=use_openai,
        require_openai=require_openai,
        top_industries_n=max(industry_limit, 20),
        top_obligors_n=max(obligor_limit, 20),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        docx_name = scenario_report_filename(scenario, extension="docx")
        docx_path = tmp_path / docx_name

        render_scenario_report_docx(
            report_payload=report_payload,
            output_path=docx_path,
            industry_limit=industry_limit,
            obligor_limit=obligor_limit,
        )

        if file_format == "docx":
            data = docx_path.read_bytes()
            return send_file(
                BytesIO(data),
                mimetype=DOCX_MIMETYPE,
                as_attachment=True,
                download_name=docx_name,
            )

        pdf_path = convert_docx_to_pdf(docx_path, output_dir=tmp_path)
        pdf_name = scenario_report_filename(scenario, extension="pdf")
        data = pdf_path.read_bytes()
        return send_file(
            BytesIO(data),
            mimetype=PDF_MIMETYPE,
            as_attachment=True,
            download_name=pdf_name,
        )

@app.route("/credit_memo", methods=["GET", "POST"])
def credit_memo():
    payload = request.get_json(silent=True) or {}

    symbol = (
        payload.get("symbol")
        or request.args.get("symbol")
        or request.args.get("ticker")
    )

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    # Everything that is not a control parameter can be treated as credit request data.
    control_keys = {"symbol", "ticker", "use_openai", "require_openai", "model", "model_tier", "model_size", "context_mode", "policy_mode", "prompt_mode", "experiment_id", "include_llm_context"}
    credit_request = {
        k: v for k, v in payload.items()
        if k not in control_keys
    }

    # GET convenience parameters for Bubble/local testing.
    for key in [
        "request_type",
        "existing_exposure_usd",
        "requested_increase_usd",
        "proposed_exposure_usd",
        "facility_type",
        "purpose",
        "tenor_years",
        "secured",
        "seniority",
        "relationship_context",
        "currency",
        "lgd",
    ]:
        if request.args.get(key) is not None:
            credit_request[key] = request.args.get(key)

    use_openai = _truthy(payload.get("use_openai", request.args.get("use_openai")), True)
    require_openai = _truthy(payload.get("require_openai", request.args.get("require_openai")), False)
    model = payload.get("model") or request.args.get("model")
    model_tier = (
        payload.get("model_tier")
        or payload.get("model_size")
        or request.args.get("model_tier")
        or request.args.get("model_size")
        or "mini"
    )
    context_mode = payload.get("context_mode") or request.args.get("context_mode", "full")
    policy_mode = payload.get("policy_mode") or request.args.get("policy_mode", "deterministic_evaluated")
    prompt_mode = payload.get("prompt_mode") or request.args.get("prompt_mode", "tight")
    experiment_id = payload.get("experiment_id") or request.args.get("experiment_id")
    include_llm_context = _truthy(
        payload.get("include_llm_context", request.args.get("include_llm_context")),
        True,
    )

    try:
        result = create_credit_memo(
            symbol=symbol,
            credit_request=credit_request,
            use_openai=use_openai,
            require_openai=require_openai,
            model=model,
            model_tier=model_tier,
            context_mode=context_mode,
            policy_mode=policy_mode,
            prompt_mode=prompt_mode,
            experiment_id=experiment_id,
            include_llm_context=include_llm_context,
        )
        return jsonify(result)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc), "type": type(exc).__name__}), 500


@app.route("/credit_memo_health")
def credit_memo_health():
    try:
        df = load_rating_context()
        return jsonify({
            "ok": True,
            "rows": int(len(df)),
            "columns": list(df.columns),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "type": type(exc).__name__}), 500

@app.route("/credit_policy")
def credit_policy_endpoint():
    try:
        policy = load_credit_policy()
        return jsonify(policy)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "type": type(exc).__name__}), 500


@app.route("/credit_policy_health")
def credit_policy_health():
    try:
        policy = load_credit_policy()
        manual = policy.get("policy_manual", {})
        return jsonify({
            "ok": True,
            "id": manual.get("id"),
            "version": manual.get("version"),
            "rules": len(policy.get("policy_rules", [])),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "type": type(exc).__name__}), 500



@app.route("/credit_memo_ablation_config")
def credit_memo_ablation_config():
    return jsonify({
        "context_modes": [
            {"value": "full", "label": "Full deterministic context visible to LLM"},
            {"value": "minimal", "label": "Minimal borrower/request context only"},
        ],
        "policy_modes": [
            {
                "value": "deterministic_evaluated",
                "label": "Deterministic evaluated",
                "description": "LLM sees the machine-readable policy manual and the backend's deterministic policy evaluation."
            },
            {
                "value": "llm_evaluated",
                "label": "LLM evaluated",
                "description": "LLM sees the machine-readable policy manual, but not the backend policy evaluation; it must apply the rules itself."
            },
            {
                "value": "none",
                "label": "None",
                "description": "LLM sees neither the policy manual nor the backend policy evaluation."
            },
        ],
        "prompt_modes": [
            {"value": "tight", "label": "Tight controlled prompt"},
            {"value": "loose", "label": "Short loose prompt"},
        ],
        "model_tiers": [
            {
                "value": "mini",
                "label": "Mini GPT model",
                "description": "Lower-cost GPT model for high-volume benchmark generation."
            },
            {
                "value": "full",
                "label": "Full GPT model",
                "description": "Higher-capability GPT model for policy reasoning and premium comparison."
            },
        ],
        "example": {
            "symbol": "0008.HK",
            "requested_increase_usd": 100000000,
            "proposed_exposure_usd": 100000000,
            "facility_type": "revolving_credit_facility",
            "purpose": "general_corporate_purposes",
            "context_mode": "minimal",
            "policy_mode": "none",
            "prompt_mode": "loose",
            "model_tier": "mini",
        },
    })

@app.route("/credit_memo_file", methods=["GET", "POST"])
def credit_memo_file_endpoint():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        source = body
        credit_request = body.get("credit_request") or _credit_request_from_source(body)
    else:
        source = request.args
        credit_request = _credit_request_from_source(source)

    symbol = source.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    file_format = str(source.get("format", "docx")).lower().strip()
    if file_format not in {"docx", "pdf"}:
        return jsonify({"error": "format must be 'docx' or 'pdf'"}), 400

    use_openai = _bool_param(source.get("use_openai"), True)
    require_openai = _bool_param(source.get("require_openai"), False)
    model = source.get("model") or None
    model_tier = source.get("model_tier") or source.get("model_size") or "mini"
    context_mode = source.get("context_mode") or "full"
    policy_mode = source.get("policy_mode") or "deterministic_evaluated"
    prompt_mode = source.get("prompt_mode") or "tight"
    experiment_id = source.get("experiment_id") or None

    try:
        memo_payload = create_credit_memo(
            symbol=symbol,
            credit_request=credit_request,
            use_openai=use_openai,
            require_openai=require_openai,
            model=model,
            model_tier=model_tier,
            context_mode=context_mode,
            policy_mode=policy_mode,
            prompt_mode=prompt_mode,
            experiment_id=experiment_id,
            include_llm_context=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            borrower = memo_payload.get("memo_context", {}).get("borrower", {})
            docx_name = credit_memo_filename(borrower, extension="docx")
            docx_path = tmp_dir / docx_name
            render_credit_memo_docx(memo_payload, docx_path)

            if file_format == "docx":
                output_path = docx_path
                download_name = docx_name
                mimetype = DOCX_MIMETYPE
            else:
                pdf_path = convert_docx_to_pdf(docx_path, output_dir=tmp_dir)
                output_path = pdf_path
                download_name = docx_name.replace(".docx", ".pdf")
                mimetype = PDF_MIMETYPE

            # Read into memory so TemporaryDirectory can safely clean up before response completes.
            data = io.BytesIO(output_path.read_bytes())
            data.seek(0)
            return send_file(
                data,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_name,
            )

    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        app.logger.exception("credit_memo_file failed")
        return jsonify({"error": "credit_memo_file failed", "detail": str(exc)}), 500


@app.route("/credit_memo_annotation_config")
def credit_memo_annotation_config():
    return jsonify(annotation_config())


@app.route("/credit_memo_annotation", methods=["GET", "POST"])
def credit_memo_annotation_endpoint():
    """
    Generate and annotate a credit memo, or annotate a supplied memo_payload.

    POST options:
      1. {"memo_payload": <payload returned by /credit_memo>}
      2. <payload returned by /credit_memo> directly
      3. normal credit_memo request fields, in which case this endpoint generates
         the memo first and then annotates it.

    GET uses the same query parameters as /credit_memo.
    """
    payload = request.get_json(silent=True) or {}

    include_memo_payload = _truthy(
        payload.get("include_memo_payload", request.args.get("include_memo_payload")),
        False,
    )

    try:
        # Case 1: explicit memo_payload wrapper.
        if isinstance(payload.get("memo_payload"), dict):
            memo_payload = payload["memo_payload"]
            annotation = annotate_credit_memo(memo_payload)
            if include_memo_payload:
                annotation["memo_payload"] = memo_payload
            return jsonify(annotation)

        # Case 2: raw /credit_memo result posted directly.
        if isinstance(payload.get("memo_context"), dict) and isinstance(payload.get("narrative"), dict):
            memo_payload = payload
            annotation = annotate_credit_memo(memo_payload)
            if include_memo_payload:
                annotation["memo_payload"] = memo_payload
            return jsonify(annotation)

        # Case 3: generate memo first, then annotate.
        symbol = (
            payload.get("symbol")
            or request.args.get("symbol")
            or request.args.get("ticker")
        )
        if not symbol:
            return jsonify({"error": "symbol is required, unless posting memo_payload"}), 400

        control_keys = {
            "symbol", "ticker", "use_openai", "require_openai", "model",
            "model_tier", "model_size", "context_mode", "policy_mode", "prompt_mode",
            "experiment_id", "include_llm_context", "include_memo_payload",
        }
        credit_request = {k: v for k, v in payload.items() if k not in control_keys}

        for key in [
            "request_type",
            "existing_exposure_usd",
            "requested_increase_usd",
            "proposed_exposure_usd",
            "facility_type",
            "purpose",
            "tenor_years",
            "secured",
            "seniority",
            "relationship_context",
            "currency",
            "lgd",
        ]:
            if request.args.get(key) is not None:
                credit_request[key] = request.args.get(key)

        use_openai = _truthy(payload.get("use_openai", request.args.get("use_openai")), True)
        require_openai = _truthy(payload.get("require_openai", request.args.get("require_openai")), True)
        model = payload.get("model") or request.args.get("model")
        model_tier = (
            payload.get("model_tier")
            or payload.get("model_size")
            or request.args.get("model_tier")
            or request.args.get("model_size")
            or "mini"
        )
        context_mode = payload.get("context_mode") or request.args.get("context_mode", "full")
        policy_mode = payload.get("policy_mode") or request.args.get("policy_mode", "deterministic_evaluated")
        prompt_mode = payload.get("prompt_mode") or request.args.get("prompt_mode", "tight")
        experiment_id = payload.get("experiment_id") or request.args.get("experiment_id")

        memo_payload = create_credit_memo(
            symbol=symbol,
            credit_request=credit_request,
            use_openai=use_openai,
            require_openai=require_openai,
            model=model,
            model_tier=model_tier,
            context_mode=context_mode,
            policy_mode=policy_mode,
            prompt_mode=prompt_mode,
            experiment_id=experiment_id,
            include_llm_context=True,
        )

        annotation = annotate_credit_memo(memo_payload)
        if include_memo_payload:
            annotation["memo_payload"] = memo_payload
        return jsonify(annotation)

    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        app.logger.exception("credit_memo_annotation failed")
        return jsonify({"error": "credit_memo_annotation failed", "detail": str(exc), "type": type(exc).__name__}), 500

@app.get("/benchmark_credit_memos")
def benchmark_credit_memos():
    """
    List stored benchmark memos.

    Optional query parameters:
        symbol
        context_mode
        policy_mode
        prompt_mode
        model_tier
        limit
    """
    try:
        rows = read_benchmark_index()
    except (RuntimeError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 500

    supported_filters = (
        "symbol",
        "context_mode",
        "policy_mode",
        "prompt_mode",
        "model_tier",
    )

    for field in supported_filters:
        requested_value = request.args.get(field)

        if requested_value:
            requested_value = requested_value.strip().lower()

            rows = [
                row
                for row in rows
                if str(row.get(field, "")).strip().lower()
                == requested_value
            ]

    try:
        limit = int(request.args.get("limit", 100))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    if limit < 1:
        return jsonify({"error": "limit must be at least 1"}), 400

    limit = min(limit, 500)

    items = [normalise_index_row(row) for row in rows[:limit]]

    return jsonify(
        {
            "count": len(items),
            "total_matching": len(rows),
            "limit": limit,
            "items": items,
        }
    )


@app.get("/benchmark_credit_memo/<memo_id>")
def benchmark_credit_memo_detail(memo_id: str):
    """
    Return the complete page model for one stored benchmark memo.

    The response includes:
        - summary and configuration metadata
        - generated memo content
        - annotation findings
        - deterministic and LLM context
        - file availability
        - related endpoint URLs
    """
    try:
        row = find_benchmark_row(memo_id)
    except (RuntimeError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 500

    if row is None:
        abort(404, description="Benchmark memo not found")

    try:
        benchmark_dir = get_benchmark_dir()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    memo_file = (
        benchmark_dir
        / "raw_memos"
        / f"{memo_id}.json"
    )

    annotation_file = (
        benchmark_dir
        / "offline_annotations"
        / f"{memo_id}__annotation.json"
    )

    try:
        memo_payload = read_json_file(
            memo_file,
            "raw memo JSON file",
        )
    except FileNotFoundError as exc:
        abort(404, description=str(exc))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    annotation_payload = None

    if annotation_file.is_file():
        try:
            annotation_payload = read_json_file(
                annotation_file,
                "annotation JSON file",
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

    index = normalise_index_row(row)
    annotation_items = extract_annotation_items(annotation_payload)

    memo = {
        "narrative": memo_payload.get("narrative"),
        "memo_markdown": memo_payload.get("memo_markdown"),
        "narrative_source": memo_payload.get("narrative_source"),
        "fallback_reason": memo_payload.get("fallback_reason"),
    }

    context = {
        "experiment_config": memo_payload.get("experiment_config"),
        "memo_context": memo_payload.get("memo_context"),
        "llm_context": memo_payload.get("llm_context"),
        "benchmark_runner": memo_payload.get("benchmark_runner"),
    }

    return jsonify(
        {
            "memo_id": memo_id,

            # Header, scores and configuration for the detail page.
            "summary": index,

            # Display-ready generated memo.
            "memo": memo,

            # Complete annotation object for scores and additional metadata.
            "annotation_summary": annotation_payload,

            # Inputs and supporting context.
            "context": context,

            # Convenience fields for buttons and file controls.
            "files": index.get("files", {}),
            "urls": index.get("urls", {}),
        }
    )

@app.get("/benchmark_credit_memo_context/<memo_id>")
def benchmark_credit_memo_context(memo_id: str):
    """
    Return the input and deterministic context for one benchmark memo.

    Retained as a lightweight specialist endpoint even though the main
    detail endpoint now includes the same context in its page model.
    """
    try:
        row = find_benchmark_row(memo_id)
    except (RuntimeError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 500

    if row is None:
        abort(404, description="Benchmark memo not found")

    try:
        benchmark_dir = get_benchmark_dir()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    memo_file = (
        benchmark_dir
        / "raw_memos"
        / f"{memo_id}.json"
    )

    try:
        memo_payload = read_json_file(
            memo_file,
            "raw memo JSON file",
        )
    except FileNotFoundError as exc:
        abort(404, description=str(exc))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "memo_id": memo_id,
            "summary": normalise_index_row(row),
            "experiment_config": memo_payload.get("experiment_config"),
            "llm_context": memo_payload.get("llm_context"),
            "memo_context": memo_payload.get("memo_context"),
            "benchmark_runner": memo_payload.get("benchmark_runner"),
        }
    )


@app.get("/benchmark_credit_memo_file/<memo_id>")
def benchmark_credit_memo_file(memo_id: str):
    """
    Download or display a rendered memo file.

    Examples:
        /benchmark_credit_memo_file/<memo_id>?format=pdf
        /benchmark_credit_memo_file/<memo_id>?format=docx
    """
    row = find_benchmark_row(memo_id)

    if row is None:
        abort(404, description="Benchmark memo not found")

    requested_format = request.args.get("format", "pdf").strip().lower()

    allowed_formats = {
        "pdf": {
            "directory": "pdf",
            "mimetype": "application/pdf",
            "as_attachment": False,
        },
        "docx": {
            "directory": "docx",
            "mimetype": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            "as_attachment": True,
        },
    }

    if requested_format not in allowed_formats:
        return jsonify(
            {"error": "format must be either pdf or docx"}
        ), 400

    format_config = allowed_formats[requested_format]
    benchmark_dir = get_benchmark_dir()

    file_path = (
        benchmark_dir
        / "rendered_files"
        / format_config["directory"]
        / f"{memo_id}.{requested_format}"
    )

    if not file_path.is_file():
        abort(
            404,
            description=f"Rendered {requested_format.upper()} file not found",
        )

    return send_file(
        file_path,
        mimetype=format_config["mimetype"],
        as_attachment=format_config["as_attachment"],
        download_name=file_path.name,
    )

@app.get("/benchmark_credit_memo_filters")
def benchmark_credit_memo_filters():
    """Return lightweight filter metadata for the benchmark explorer."""

    return jsonify(
        {
            "companies": BENCHMARK_COMPANIES,
            "models": BENCHMARK_MODELS,
            "contexts": BENCHMARK_CONTEXTS,
            "evaluation_modes": BENCHMARK_EVALUATION_MODES,
            "prompt_modes": BENCHMARK_PROMPT_MODES,
        }
    )

@app.get("/benchmark_credit_memo_html_file/<memo_id>")
def benchmark_credit_memo_html_file(memo_id: str):
    html_path = resolve_credit_memo_html_file(memo_id)

    return send_file(
        html_path,
        mimetype="text/html",
        as_attachment=False,
        conditional=True,
        max_age=3600,
    )

@app.get("/benchmark_credit_memo_html/<memo_id>")
def benchmark_credit_memo_html(memo_id: str):
    html_path = resolve_credit_memo_html_file(memo_id)

    try:
        standalone_html = html_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        abort(500, description="Stored HTML file is not valid UTF-8.")
    except OSError as exc:
        abort(500, description=f"Could not read stored HTML: {exc}")

    soup = BeautifulSoup(standalone_html, "html.parser")

    memo_shell = soup.select_one(".cb-preview-shell")

    if memo_shell is None:
        abort(
            500,
            description="Stored HTML does not contain .cb-preview-shell.",
        )

    memo_html = memo_shell.decode_contents()

    # Copy styles from the standalone document.
    style_tags = soup.find_all("style")
    styles = "\n".join(str(tag) for tag in style_tags)

    return jsonify(
        {
            "memo_id": memo_id,
            "html_filename": html_path.name,
            "memo_html": f"{styles}\n{memo_html}",
        }
    )


@app.get("/html_sections/<memo_id>")
def html_sections(memo_id: str):
    html_path = resolve_credit_memo_html_file(memo_id)

    try:
        standalone_html = html_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        abort(500, description="Stored HTML file is not valid UTF-8.")
    except OSError as exc:
        abort(500, description=f"Could not read stored HTML: {exc}")

    soup = BeautifulSoup(standalone_html, "html.parser")

    #
    # Extract all stylesheet blocks
    #
    css = "\n".join(
        style.decode_contents()
        for style in soup.find_all("style")
    )

    css_html = f"<style>\n{css}\n</style>"

    #
    # Extract memo sections
    #
    section_elements = soup.find_all("section")

    sections = []

    for index, section in enumerate(section_elements, start=1):

        heading = section.find(["h1", "h2", "h3", "h4", "h5", "h6"])

        sections.append({
            "section_id": (
                section.get("data-section-id")
                or section.get("id")
                or f"section_{index:03d}"
            ),
            "section_type": section.get("data-section-type", "other"),
            "section_title": (
                heading.get_text(strip=True)
                if heading
                else ""
            ),
            "html": str(section)
        })

    return jsonify({
        "memo_id": memo_id,
        "html_filename": html_path.name,
        "css": css_html,
        "section_count": len(sections),
        "sections": sections
    })


if __name__ == "__main__":
    app.run(debug=True)