from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required = [
        "study_hours",
        "attendance",
        "previous_score",
        "sleep_hours"
    ]

    if not data or any(key not in data for key in required):
        return jsonify({"error": "Provide all four input values"}), 400

    try:
        values = pd.DataFrame([{
            key: float(data[key]) for key in required
        }])

        if not all(values.iloc[0].map(lambda x: x == x)):
            return jsonify({"error": "Inputs must be valid numbers"}), 400

        score = float(model.predict(values)[0])

        return jsonify({
            "predicted_score": round(score, 2)
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Inputs must be numbers"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
