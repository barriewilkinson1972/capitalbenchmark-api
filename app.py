from flask import Flask, request, jsonify
from flask_cors import CORS
from model.stress_model import run_stress

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
    technology = float(request.args.get("technology", 0))
    commodity = float(request.args.get("commodity", 0))

    lgd = float(request.args.get("lgd", 0.45))
    n_sims = int(request.args.get("n_sims", 10_000))
    n_bins = int(request.args.get("n_bins", 40))
    asset_rho = float(request.args.get("asset_rho", 0.20))
    random_seed = int(request.args.get("random_seed", 42))

    result = run_stress(
        market=market,
        technology=technology,
        commodity=commodity,
        lgd=lgd,
        n_sims=n_sims,
        n_bins=n_bins,
        asset_rho=asset_rho,
        random_seed=random_seed,
    )

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)