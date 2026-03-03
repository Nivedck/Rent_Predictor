import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")
PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
CITIES_PATH = os.path.join(MODEL_DIR, "cities.json")


def load_model():
    if os.path.exists(PIPE_PATH):
        return joblib.load(PIPE_PATH)
    return None


def load_cities():
    if os.path.exists(CITIES_PATH):
        with open(CITIES_PATH, "r") as f:
            return json.load(f)
    return ["Kolkata", "Mumbai", "Delhi", "Bengaluru", "Chennai"]


pipeline = load_model()
cities = load_cities()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", cities=cities)


@app.route("/predict", methods=["POST"])
def predict():
    error = None
    prediction = None

    if pipeline is None:
        error = "Model not found. Please run train.py to create the model first."
    else:
        try:
            bhk = int(request.form.get("bhk", 2))
            size = int(request.form.get("size", 800))
            city = request.form.get("city", cities[0])
            X = pd.DataFrame([{"BHK": bhk, "Size": size, "City": city}])
            pred = pipeline.predict(X)[0]
            prediction = max(0, float(pred))
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        cities=cities,
        prediction=prediction,
        error=error,
        bhk=request.form.get("bhk", 2),
        size=request.form.get("size", 800),
        city=request.form.get("city", cities[0] if cities else ""),
    )


if __name__ == "__main__":
    app.run(debug=True)
