from flask import Flask, request, jsonify
from flask_cors import CORS

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
            {"name": "Current market", "oil": -1.5, "ai": 3.0},
            {"name": "AI normalises", "oil": -1.5, "ai": 0.0},
            {"name": "AI bust", "oil": -1.5, "ai": -1.5},
            {"name": "Oil shock only", "oil": -1.5, "ai": 0.0},
            {"name": "AI shock only", "oil": 0.0, "ai": -1.5}
        ]
    })


@app.route("/stress")
def stress():
    oil = float(request.args.get("oil", 0))
    ai = float(request.args.get("ai", 0))

    # Placeholder response for Bubble testing.
    # Later we replace this with the real model output.
    portfolio_pd = 0.005 + max(0, -oil) * 0.01 + max(0, -ai) * 0.015

    return jsonify({
        "oil": oil,
        "ai": ai,
        "portfolio_pd": portfolio_pd,
        "portfolio_pd_percent": portfolio_pd * 100,
        "expected_loss": portfolio_pd * 100_000_000 * 0.45,
        "top_industries": [
            {
                "industry": "Airlines",
                "base_pd": 0.008,
                "stressed_pd": portfolio_pd * 1.8,
                "change": portfolio_pd * 1.8 - 0.008
            },
            {
                "industry": "Auto Manufacturers",
                "base_pd": 0.006,
                "stressed_pd": portfolio_pd * 1.5,
                "change": portfolio_pd * 1.5 - 0.006
            },
            {
                "industry": "Banks - Diversified",
                "base_pd": 0.005,
                "stressed_pd": portfolio_pd * 1.3,
                "change": portfolio_pd * 1.3 - 0.005
            }
        ]
    })


if __name__ == "__main__":
    app.run(debug=True)