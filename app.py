from flask import Flask, request, jsonify
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
from model.credit_memo import create_credit_memo, load_rating_context
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
from flask import send_file, request, jsonify

app = Flask(__name__)

# Prototype: allow Bubble / browser calls
CORS(app)

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
    control_keys = {"symbol", "ticker", "use_openai", "require_openai", "model"}
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

    try:
        result = create_credit_memo(
            symbol=symbol,
            credit_request=credit_request,
            use_openai=use_openai,
            require_openai=require_openai,
            model=model,
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

    try:
        memo_payload = create_credit_memo(
            symbol=symbol,
            credit_request=credit_request,
            use_openai=use_openai,
            require_openai=require_openai,
            model=model,
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


if __name__ == "__main__":
    app.run(debug=True)