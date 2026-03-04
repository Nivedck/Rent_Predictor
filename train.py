import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib


def load_data(paths):
    dfs = []
    for p in paths:
        if os.path.exists(p):
            try:
                dfs.append(pd.read_csv(p))
            except Exception:
                pass
    if not dfs:
        raise FileNotFoundError("No dataset files found")
    df = pd.concat(dfs, ignore_index=True)
    return df


KEEP_COLS = [
    "BHK", "Size", "City", "Bathroom",
    "Area Type", "Furnishing Status", "Tenant Preferred", "Point of Contact",
    "Rent",
]


def prepare(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for required in KEEP_COLS:
        if required not in df.columns:
            raise KeyError(f"Required column '{required}' not found in data")
    df = df[KEEP_COLS]
    df = df.dropna()
    for num_col in ("BHK", "Size", "Bathroom", "Rent"):
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    df = df.dropna()
    df = df[(df["Size"] > 20) & (df["BHK"] > 0) & (df["Rent"] > 0) & (df["Rent"] < 1_000_000)]
    df["City"] = df["City"].astype(str).str.title().str.strip()
    for cat_col in ("Area Type", "Furnishing Status", "Tenant Preferred", "Point of Contact"):
        df[cat_col] = df[cat_col].astype(str).str.strip()
    return df


def train_and_save(df, out_dir="model"):
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = ["BHK", "Size", "Bathroom",
                    "City", "Area Type", "Furnishing Status",
                    "Tenant Preferred", "Point of Contact"]
    X = df[feature_cols]
    y = df["Rent"]

    numeric_features = ["BHK", "Size", "Bathroom"]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_features = ["City", "Area Type", "Furnishing Status",
                            "Tenant Preferred", "Point of Contact"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    pipeline.fit(X, y)

    joblib.dump(pipeline, os.path.join(out_dir, "pipeline.pkl"))

    # Save category options for each categorical feature
    categories = {}
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    for i, col_name in enumerate(categorical_features):
        try:
            cats = sorted(cat_encoder.categories_[i].tolist())
        except Exception:
            cats = sorted(df[col_name].unique().tolist())
        categories[col_name] = cats
    with open(os.path.join(out_dir, "categories.json"), "w") as f:
        json.dump(categories, f, indent=2)

    print("Model saved to:", os.path.join(out_dir, "pipeline.pkl"))


def main():
    base = os.path.dirname(__file__)
    paths = [
        os.path.join(base, "rent2.csv"),
    ]
    df = load_data(paths)
    df = prepare(df)
    train_and_save(df)


if __name__ == "__main__":
    main()
