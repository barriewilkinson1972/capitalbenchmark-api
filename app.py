from flask import Flask, request, jsonify
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


if __name__ == "__main__":
    app.run(debug=True)