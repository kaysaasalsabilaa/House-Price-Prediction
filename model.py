import joblib
import pandas as pd
import numpy as np

FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

def load_model(path="best_model.pkl"):
    return joblib.load(path)

def prepare_input(model, X_input: pd.DataFrame) -> pd.DataFrame:
    X = X_input.copy()

    if "MEDV" in X.columns:
        X = X.drop(columns=["MEDV"])

    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Kolom kurang: {missing}. Harusnya ada: {FEATURES}")

    X = X[FEATURES]

    X["CHAS"] = X["CHAS"].astype(int)
    X["RAD"] = X["RAD"].astype(int)
    X["TAX"] = X["TAX"].astype(int)

    X["B"] = np.log1p(X["B"])
    X["CRIM"] = np.log1p(X["CRIM"])

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_)

    return X

def predict_price(model, X_input: pd.DataFrame):
    X_ready = prepare_input(model, X_input)
    return model.predict(X_ready)
