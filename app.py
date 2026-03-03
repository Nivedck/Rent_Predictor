import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")
PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.json")


def load_model():
    if os.path.exists(PIPE_PATH):
        return joblib.load(PIPE_PATH)
    return None


def load_categories():
    if os.path.exists(CATEGORIES_PATH):
        with open(CATEGORIES_PATH, "r") as f:
            return json.load(f)
    return {
        "City": ["Kolkata", "Mumbai", "Delhi", "Bengaluru", "Chennai"],
        "Area Type": ["Super Area", "Carpet Area", "Built Area"],
        "Furnishing Status": ["Unfurnished", "Semi-Furnished", "Furnished"],
        "Tenant Preferred": ["Bachelors", "Bachelors/Family", "Family"],
        "Point of Contact": ["Contact Owner", "Contact Agent"],
    }


pipeline = load_model()
categories = load_categories()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", categories=categories)


@app.route("/predict", methods=["POST"])
def predict():
    error = None
    prediction = None
    form = request.form

    if pipeline is None:
        error = "Model not found. Please run train.py to create the model first."
    else:
        try:
            bhk = int(form.get("bhk", 2))
            size = int(form.get("size", 800))
            bathroom = int(form.get("bathroom", 1))
            city = form.get("city", categories["City"][0])
            area_type = form.get("area_type", categories["Area Type"][0])
            furnishing = form.get("furnishing", categories["Furnishing Status"][0])
            tenant = form.get("tenant", categories["Tenant Preferred"][0])
            contact = form.get("contact", categories["Point of Contact"][0])

            X = pd.DataFrame([{
                "BHK": bhk,
                "Size": size,
                "Bathroom": bathroom,
                "City": city,
                "Area Type": area_type,
                "Furnishing Status": furnishing,
                "Tenant Preferred": tenant,
                "Point of Contact": contact,
            }])
            pred = pipeline.predict(X)[0]
            prediction = max(0, float(pred))
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        categories=categories,
        prediction=prediction,
        error=error,
        form=form,
    )


if __name__ == "__main__":
    app.run(debug=True)
