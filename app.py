import os
import json
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Rent Predictor", layout="centered")

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
    # fallback: a few common cities
    return ["Kolkata", "Mumbai", "Delhi", "Bengaluru", "Chennai"]


pipeline = load_model()
cities = load_cities()

st.markdown("""
<style>
    :root {
        --bg-1: #0b1220;
        --bg-2: #0f172a;
        --card: #111827;
        --muted: #94a3b8;
        --text: #e5e7eb;
        --accent: #38bdf8;
    }

    html, body, .stApp, .block-container, .main {
        background: linear-gradient(180deg, var(--bg-1), var(--bg-2)) !important;
        color: var(--text) !important;
    }

    h1, h2, h3, p, label, span, div, a {
        color: var(--text) !important;
    }

    .stCaption, .stMarkdown p {
        color: var(--muted) !important;
    }

    .stForm {
        background: var(--card) !important;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 16px;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background: #0b1220 !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
    }

    .stButton button {
        background: linear-gradient(90deg, #0ea5e9, #38bdf8) !important;
        color: #0b1220 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    .big-number { font-size:28px; font-weight:700; color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

st.title("Room Rent Predictor")
st.caption("Predict monthly rent from BHK, Size (sqft) and City — linear regression model")

with st.form("input_form"):
    col1, col2 = st.columns([1, 2])
    with col1:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
        size = st.number_input("Size (sqft)", min_value=20, max_value=10000, value=800, step=10)
    with col2:
        city = st.selectbox("City", options=cities, index=0)
        st.write("\n")

    submitted = st.form_submit_button("Predict Rent")

if submitted:
    if pipeline is None:
        st.error("Model not found. Please run `train.py` to create the model first.")
    else:
        X = pd.DataFrame([{"BHK": bhk, "Size": size, "City": city}])
        pred = pipeline.predict(X)[0]
        pred = max(0, float(pred))
        st.markdown(f"**Predicted Monthly Rent:** <span class='big-number'>₹ {pred:,.0f}</span>", unsafe_allow_html=True)
        st.metric(label="Estimated Rent", value=f"₹ {pred:,.0f}")

        st.info("This prediction uses a simple linear regression trained on the dataset in the repo.")
