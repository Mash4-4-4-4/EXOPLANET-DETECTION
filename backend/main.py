from pathlib import Path
from typing import Any
import io
import math

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    __import__("os").environ.get(
        "EXO_MODEL_PATH",
        str(BASE_DIR / "models" / "exo_model_final.pkl"),
    )
)

MAX_UPLOAD_MB = 50
MAX_ROWS = 100_000

RAW_FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_incl",
    "koi_prad",
    "koi_sma",
    "koi_ror",
    "koi_dor",
    "koi_teq",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
    "koi_num_transits",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_model_snr",
    "koi_insol",
    "koi_srho",
]

ERROR_SOURCES = [
    "koi_period_err1",
    "koi_prad_err1",
    "koi_duration_err1",
    "koi_depth_err1",
]

ENGINEERED_FEATURES = [
    "duration_period_ratio",
    "stellar_density_calculated",
    "radius_mass_ratio",
    "planet_star_radius_ratio",
    "snr_per_transit",
    "depth_ror_interaction",
    "impact_inclination_interaction",
    "snr_transit_count",
    "depth_ror_ratio",
    "semi_major_stellar_radius_ratio",
]

CLASS_NAMES = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]

app = FastAPI(
    title="EXO DETECTOR API",
    version="V9-FINAL",
    description="Kepler KOI exoplanet classification API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_package() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Final model not found at {MODEL_PATH}. "
            "Copy exo_model_v9.pkl to backend/models/exo_model_final.pkl."
        )

    package = joblib.load(MODEL_PATH)

    required = {"models", "meta_model", "features", "label_encoder"}
    missing = required.difference(package.keys())
    if missing:
        raise RuntimeError(f"Invalid model package. Missing: {sorted(missing)}")

    return package


MODEL = None


@app.on_event("startup")
def startup() -> None:
    global MODEL
    MODEL = load_package()
    print(f"Loaded V9 final model from: {MODEL_PATH}")
    print(f"Features: {len(MODEL['features'])}")
    print(f"Model families: {MODEL['model_families']}")


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if column not in {"kepid", "koi_disposition", "kepoi_name"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing_raw = [c for c in RAW_FEATURES if c not in df.columns]
    if missing_raw:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CSV is missing required Kepler feature columns.",
                "missing_columns": missing_raw,
            },
        )

    work = df.copy()

    work["duration_period_ratio"] = (
        work["koi_duration"] / (work["koi_period"] + 1e-6)
    )
    work["stellar_density_calculated"] = (
        work["koi_smass"] / ((work["koi_srad"] + 1e-6) ** 3)
    )
    work["radius_mass_ratio"] = (
        work["koi_prad"] / (work["koi_smass"] + 1e-6)
    )
    work["planet_star_radius_ratio"] = (
        work["koi_prad"] / (work["koi_srad"] + 1e-6)
    )
    work["snr_per_transit"] = (
        work["koi_model_snr"] / (work["koi_num_transits"] + 1e-6)
    )
    work["depth_ror_interaction"] = (
        work["koi_depth"] * work["koi_ror"]
    )
    work["impact_inclination_interaction"] = (
        work["koi_impact"] * work["koi_incl"]
    )
    work["snr_transit_count"] = (
        work["koi_model_snr"] * work["koi_num_transits"]
    )
    work["depth_ror_ratio"] = (
        work["koi_depth"] / work["koi_ror"].replace(0, np.nan)
    )
    work["semi_major_stellar_radius_ratio"] = (
        work["koi_sma"] / work["koi_srad"].replace(0, np.nan)
    )

    if "koi_period_err1" in work.columns:
        work["period_rel_err"] = (
            work["koi_period_err1"].abs()
            / (work["koi_period"].abs() + 1e-6)
        )
    if "koi_prad_err1" in work.columns:
        work["prad_rel_err"] = (
            work["koi_prad_err1"].abs()
            / (work["koi_prad"].abs() + 1e-6)
        )
    if "koi_duration_err1" in work.columns:
        work["duration_rel_err"] = (
            work["koi_duration_err1"].abs()
            / (work["koi_duration"].abs() + 1e-6)
        )
    if "koi_depth_err1" in work.columns:
        work["depth_rel_err"] = (
            work["koi_depth_err1"].abs()
            / (work["koi_depth"].abs() + 1e-6)
        )

    for column in features:
        if column not in work.columns:
            work[column] = np.nan

    return work[features].replace([np.inf, -np.inf], np.nan)


def family_probabilities(X: pd.DataFrame) -> np.ndarray:
    package = MODEL
    family_probs = []

    for family_name in package["model_families"]:
        models = package["models"][family_name]
        if not models:
            raise RuntimeError(f"No trained models found for family: {family_name}")

        family_sum = None
        for model in models:
            probabilities = model.predict_proba(X)
            family_sum = (
                probabilities
                if family_sum is None
                else family_sum + probabilities
            )

        family_probs.append(family_sum / len(models))

    return np.hstack(family_probs)


def json_number(value: Any):
    if value is None:
        return None
    try:
        value = float(value)
        return None if not math.isfinite(value) else value
    except (TypeError, ValueError):
        return None


def row_features(row: pd.Series) -> dict[str, Any]:
    wanted = [
        "koi_period",
        "koi_duration",
        "koi_prad",
        "koi_teq",
        "koi_srad",
        "koi_model_snr",
        "koi_depth",
        "koi_sma",
        "koi_ror",
        "koi_dor",
        "koi_num_transits",
        "koi_steff",
        "koi_smass",
    ]

    return {key: json_number(row.get(key)) for key in wanted}


@app.get("/")
def root():
    return {
        "name": "EXO DETECTOR API",
        "version": "V9-FINAL",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "version": "V9-FINAL",
        "features": len(MODEL["features"]) if MODEL else 0,
        "model_families": MODEL["model_families"] if MODEL else [],
    }


@app.get("/model-info")
def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "version": "V9-FINAL",
        "features": MODEL["features"],
        "feature_count": len(MODEL["features"]),
        "model_families": MODEL["model_families"],
        "final_seeds": MODEL.get("final_seeds", []),
        "fp_flags_used": [],
        "metrics": {
            "accuracy": 0.8485,
            "macro_f1": 0.7961,
            "weighted_f1": 0.8436,
            "balanced_accuracy": 0.7872,
            "roc_auc": 0.9352,
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    raw = await file.read()

    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_UPLOAD_MB} MB upload limit.",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw), comment="#")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV contains no rows.")

    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV contains more than {MAX_ROWS:,} rows.",
        )

    df = clean_numeric(df)

    X = build_features(df, MODEL["features"])
    stacked = family_probabilities(X)
    final_prob = MODEL["meta_model"].predict_proba(stacked)

    encoder = MODEL["label_encoder"]
    predicted_indices = np.argmax(final_prob, axis=1)
    labels = encoder.inverse_transform(predicted_indices)

    predictions = []

    for i, label in enumerate(labels):
        probabilities = {
            str(class_name): float(final_prob[i, j])
            for j, class_name in enumerate(encoder.classes_)
        }

        kepid = df.iloc[i].get("kepid")
        name = df.iloc[i].get("kepoi_name")

        if pd.isna(name) or not str(name).strip():
            name = f"KOI-{kepid}" if not pd.isna(kepid) else f"Object-{i + 1}"

        predictions.append(
            {
                "index": i,
                "name": str(name),
                "kepid": None if pd.isna(kepid) else str(kepid),
                "prediction": str(label),
                "probabilities": probabilities,
                "features": row_features(df.iloc[i]),
            }
        )

    counts = {
        "CONFIRMED": int(np.sum(labels == "CONFIRMED")),
        "CANDIDATE": int(np.sum(labels == "CANDIDATE")),
        "FALSE POSITIVE": int(np.sum(labels == "FALSE POSITIVE")),
    }

    return {
        "version": "V9-FINAL",
        "filename": file.filename,
        "rows": len(df),
        "counts": counts,
        "predictions": predictions,
    }
