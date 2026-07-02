from flask import Flask, request, jsonify
from flask_cors import CORS
from model.stress_model import run_stress

from io import BytesIO
from datetime import datetime
import zipfile

import pandas as pd
from flask import send_file, request, jsonify

app = Flask(__name__)

# Prototype: allow Bubble / browser calls
CORS(app)


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

    top_industries = pd.DataFrame(result.get("top_industries", []))[["industry", "obligors", "ead", "base_pd", "stressed_pd", "stressed_expected_loss", "pd_multiple", "rho_Market", "rho_Technology", "rho_Commodity", "x_plot", "y_plot"]]
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


if __name__ == "__main__":
    app.run(debug=True)