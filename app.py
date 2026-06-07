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
    oil = float(request.args.get("oil", 0))
    ai = float(request.args.get("ai", 0))

    result = run_stress(oil=oil, ai=ai)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)