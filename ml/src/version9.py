#without fp flags
# ============================================================
# EXOPLANET DETECTION - VERSION 7 FINAL
#
# FEATURES:
#   - 21 astronomical / observational raw features
#   - 4 NASA KOI false-positive vetting flags
#   - Relative measurement-error features
#   - Engineered astronomical features
#
# MODELS:
#   - XGBoost
#   - LightGBM
#   - Multiple random seeds
#   - OOF stacking
#   - Logistic Regression meta-learner
#
# VALIDATION:
#   - StratifiedGroupKFold
#   - Grouped by kepid
#   - No star overlap between train/test
#
# TARGET:
#   CANDIDATE
#   CONFIRMED
#   FALSE POSITIVE
# ============================================================


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# OPTIONAL LIGHTGBM
# ============================================================

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True

except ImportError:

    HAS_LGBM = False

    print(
        "WARNING: lightgbm is not installed.\n"
        "Install it using:\n"
        "pip install lightgbm"
    )


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/kepler_data.csv"

df = pd.read_csv(DATA_PATH)


print("\n==============================")
print("DATASET")
print("==============================")

print("Shape:", df.shape)
print("Unique stars:", df["kepid"].nunique())


print("\nClass distribution:")
print(df["koi_disposition"].value_counts())


# ============================================================
# 2. CLEAN DATA
# ============================================================

df = df.dropna(
    subset=[
        "koi_disposition",
        "kepid"
    ]
).copy()


# ============================================================
# 3. RAW ASTRONOMICAL FEATURES
# ============================================================

raw_features = [

    # -------------------------
    # Planet / transit features
    # -------------------------

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

    # -------------------------
    # Stellar features
    # -------------------------

    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",

    # -------------------------
    # Detection quality
    # -------------------------

    "koi_num_transits",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_model_snr",

    # -------------------------
    # Stellar irradiation /
    # density
    # -------------------------

    "koi_insol",
    "koi_srho"
]


# ============================================================
# 4. NASA FALSE-POSITIVE FLAGS
# ============================================================

fp_flag_candidates = [
]


fp_flags = [
    c for c in fp_flag_candidates
    if c in df.columns
]


if fp_flags:

    print("\nUsing FP flags:")
    print(fp_flags)

else:

    print(
        "\nWARNING: No koi_fpflag_* columns found."
    )


# ============================================================
# 5. MEASUREMENT ERROR COLUMNS
# ============================================================

err_source_cols = [

    "koi_period_err1",
    "koi_prad_err1",
    "koi_duration_err1",
    "koi_depth_err1"
]


err_cols_present = [
    c for c in err_source_cols
    if c in df.columns
]


if err_cols_present:

    print("\nUsing error columns:")
    print(err_cols_present)

else:

    print(
        "\nWARNING: No measurement-error columns found."
    )


# ============================================================
# 6. ENGINEERED FEATURES
# ============================================================


# ------------------------------------------------------------
# Transit duration / orbital period
# ------------------------------------------------------------

df["duration_period_ratio"] = (
    df["koi_duration"]
    /
    (df["koi_period"] + 1e-6)
)


# ------------------------------------------------------------
# Stellar density
# rho ~ mass / radius^3
# ------------------------------------------------------------

df["stellar_density_calculated"] = (
    df["koi_smass"]
    /
    ((df["koi_srad"] + 1e-6) ** 3)
)


# ------------------------------------------------------------
# Planet radius / stellar mass
# ------------------------------------------------------------

df["radius_mass_ratio"] = (
    df["koi_prad"]
    /
    (df["koi_smass"] + 1e-6)
)


# ------------------------------------------------------------
# Planet radius / star radius
# ------------------------------------------------------------

df["planet_star_radius_ratio"] = (
    df["koi_prad"]
    /
    (df["koi_srad"] + 1e-6)
)


# ------------------------------------------------------------
# Signal-to-noise per transit
# ------------------------------------------------------------

df["snr_per_transit"] = (
    df["koi_model_snr"]
    /
    (df["koi_num_transits"] + 1e-6)
)


# ------------------------------------------------------------
# Transit depth × radius ratio
# ------------------------------------------------------------

df["depth_ror_interaction"] = (
    df["koi_depth"]
    *
    df["koi_ror"]
)


# ------------------------------------------------------------
# Impact parameter × inclination
# ------------------------------------------------------------

df["impact_inclination_interaction"] = (
    df["koi_impact"]
    *
    df["koi_incl"]
)


# ------------------------------------------------------------
# SNR × number of transits
# ------------------------------------------------------------

df["snr_transit_count"] = (
    df["koi_model_snr"]
    *
    df["koi_num_transits"]
)


# ------------------------------------------------------------
# Transit depth / radius ratio
# ------------------------------------------------------------

df["depth_ror_ratio"] = (
    df["koi_depth"]
    /
    df["koi_ror"].replace(0, np.nan)
)


# ------------------------------------------------------------
# Semi-major axis / stellar radius
# ------------------------------------------------------------

df["semi_major_stellar_radius_ratio"] = (
    df["koi_sma"]
    /
    df["koi_srad"].replace(0, np.nan)
)


engineered_features = [

    "duration_period_ratio",

    "stellar_density_calculated",

    "radius_mass_ratio",

    "planet_star_radius_ratio",

    "snr_per_transit",

    "depth_ror_interaction",

    "impact_inclination_interaction",

    "snr_transit_count",

    "depth_ror_ratio",

    "semi_major_stellar_radius_ratio"
]


# ============================================================
# 7. RELATIVE MEASUREMENT ERROR FEATURES
# ============================================================

error_features = []


# ------------------------------------------------------------
# Period relative error
# ------------------------------------------------------------

if "koi_period_err1" in df.columns:

    df["period_rel_err"] = (
        df["koi_period_err1"].abs()
        /
        (df["koi_period"].abs() + 1e-6)
    )

    error_features.append(
        "period_rel_err"
    )


# ------------------------------------------------------------
# Planet radius relative error
# ------------------------------------------------------------

if "koi_prad_err1" in df.columns:

    df["prad_rel_err"] = (
        df["koi_prad_err1"].abs()
        /
        (df["koi_prad"].abs() + 1e-6)
    )

    error_features.append(
        "prad_rel_err"
    )


# ------------------------------------------------------------
# Duration relative error
# ------------------------------------------------------------

if "koi_duration_err1" in df.columns:

    df["duration_rel_err"] = (
        df["koi_duration_err1"].abs()
        /
        (df["koi_duration"].abs() + 1e-6)
    )

    error_features.append(
        "duration_rel_err"
    )


# ------------------------------------------------------------
# Depth relative error
# ------------------------------------------------------------

if "koi_depth_err1" in df.columns:

    df["depth_rel_err"] = (
        df["koi_depth_err1"].abs()
        /
        (df["koi_depth"].abs() + 1e-6)
    )

    error_features.append(
        "depth_rel_err"
    )


engineered_features = (
    engineered_features
    +
    error_features
)


# ============================================================
# 8. FINAL FEATURE LIST
# ============================================================

# IMPORTANT:
# V7 FINAL includes NASA's FP flags.

features = (
    raw_features
    +
    engineered_features
    +
    fp_flags
)


print("\n==============================")
print("FEATURES")
print("==============================")

print(
    "Raw features:",
    len(raw_features)
)

print(
    "Engineered features:",
    len(engineered_features)
)

print(
    "FP flag features:",
    len(fp_flags)
)

print(
    "Total features:",
    len(features)
)


# ============================================================
# 9. X / Y / GROUPS
# ============================================================

X = df[features].copy()

# Convert infinity to NaN so the imputer can handle it
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Target
y = df["koi_disposition"]


# Group
# Every observation belonging to the same star
# must stay in the same split.
groups = df["kepid"]


# ============================================================
# 10. ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

Y = encoder.fit_transform(y)


print("\nClasses:")

print(
    dict(
        zip(
            encoder.classes_,
            range(len(encoder.classes_))
        )
    )
)


# ============================================================
# 11. OUTER GROUP SPLIT
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


train_idx, test_idx = next(
    outer_cv.split(
        X,
        Y,
        groups
    )
)


X_train = (
    X.iloc[train_idx]
    .reset_index(drop=True)
)

X_test = (
    X.iloc[test_idx]
    .reset_index(drop=True)
)


Y_train = Y[train_idx]

Y_test = Y[test_idx]


groups_train = (
    groups.iloc[train_idx]
    .reset_index(drop=True)
)

groups_test = (
    groups.iloc[test_idx]
    .reset_index(drop=True)
)


print("\n==============================")
print("OUTER SPLIT")
print("==============================")


print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print(
    "Training stars:",
    groups_train.nunique()
)

print(
    "Testing stars:",
    groups_test.nunique()
)


# ============================================================
# 12. STAR LEAKAGE CHECK
# ============================================================

overlap = set(
    groups_train
).intersection(
    set(groups_test)
)


print(
    "Overlapping stars:",
    len(overlap)
)


if len(overlap) == 0:

    print(
        "NO STAR LEAKAGE"
    )

else:

    raise ValueError(
        "STAR LEAKAGE DETECTED"
    )


# ============================================================
# 13. XGBOOST MODEL
# ============================================================

def create_xgb(seed):

    return Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",

            XGBClassifier(

                objective="multi:softprob",

                num_class=3,

                eval_metric="mlogloss",

                tree_method="hist",

                n_estimators=500,

                max_depth=6,

                learning_rate=0.04,

                subsample=0.85,

                colsample_bytree=0.9,

                min_child_weight=5,

                gamma=0.15,

                reg_lambda=10,

                reg_alpha=0.2,

                max_leaves=31,

                grow_policy="depthwise",

                random_state=seed,

                n_jobs=-1
            )
        )
    ])


# ============================================================
# 14. LIGHTGBM MODEL
# ============================================================

def create_lgbm(seed):

    if not HAS_LGBM:

        return None


    return Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",

            LGBMClassifier(

                objective="multiclass",

                num_class=3,

                n_estimators=600,

                max_depth=-1,

                num_leaves=31,

                learning_rate=0.035,

                subsample=0.85,

                colsample_bytree=0.85,

                min_child_samples=20,

                reg_lambda=10,

                reg_alpha=0.2,

                random_state=seed,

                n_jobs=-1,

                verbose=-1
            )
        )
    ])


# ============================================================
# 15. MODEL FACTORIES
# ============================================================

model_factories = [

    (
        "xgb",
        create_xgb
    )
]


if HAS_LGBM:

    model_factories.append(
        (
            "lgbm",
            create_lgbm
        )
    )


# OOF seeds
seeds = [
    42,
    52,
    62
]


# ============================================================
# 16. OOF PREDICTION GENERATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)


print("\n==============================")
print("OOF GENERATION")
print("==============================")


# Store OOF probabilities
oof_probs_by_family = {

    name:
    np.zeros(
        (
            len(X_train),
            3
        )
    )

    for name, _
    in model_factories
}


# ------------------------------------------------------------
# Five grouped folds
# ------------------------------------------------------------

for fold, (
    tr_idx,
    val_idx
) in enumerate(

    cv.split(
        X_train,
        Y_train,
        groups_train
    )

):

    print(
        f"\nFold {fold + 1}/5"
    )


    # --------------------------------------------------------
    # Train every model family
    # --------------------------------------------------------

    for name, factory in model_factories:

        fold_prob = np.zeros(
            (
                len(val_idx),
                3
            )
        )


        # ----------------------------------------------------
        # Multiple seeds
        # ----------------------------------------------------

        for seed in seeds:

            model = factory(seed)


            model.fit(

                X_train.iloc[tr_idx],

                Y_train[tr_idx],

                # IMPORTANT:
                # sample weights are NOT used in final V7.
                # This preserves the original V7 behavior.
            )


            fold_prob += (
                model.predict_proba(
                    X_train.iloc[val_idx]
                )
            )


        # Average the seeds
        fold_prob /= len(seeds)


        # Save OOF probabilities
        oof_probs_by_family[
            name
        ][val_idx] = fold_prob


# ============================================================
# 17. STACKING META-LEARNER
# ============================================================

print("\n==============================")
print("STACKING META-LEARNER")
print("==============================")


# Concatenate predictions from
# XGBoost and LightGBM.

meta_X_train = np.hstack(

    [
        oof_probs_by_family[name]

        for name, _
        in model_factories
    ]

)


# ------------------------------------------------------------
# Logistic Regression meta learner
# ------------------------------------------------------------

meta_model = LogisticRegression(

    max_iter=2000,

    C=1.0
)


meta_model.fit(

    meta_X_train,

    Y_train
)


# ------------------------------------------------------------
# OOF prediction
# ------------------------------------------------------------

oof_meta_pred = (
    meta_model.predict(
        meta_X_train
    )
)


oof_meta_f1 = f1_score(

    Y_train,

    oof_meta_pred,

    average="macro"
)


print(
    "OOF Macro F1 (stacked):",
    round(
        oof_meta_f1,
        4
    )
)


# ============================================================
# 18. TRAIN FINAL BASE MODELS
# ============================================================

print("\n==============================")
print("TRAINING FINAL BASE MODELS")
print("==============================")


# More seeds for final ensemble
final_seeds = [
    42,
    52,
    62,
    72,
    82
]


# Store test probabilities
test_probs_by_family = {

    name:
    np.zeros(
        (
            len(X_test),
            3
        )
    )

    for name, _
    in model_factories
}


# ------------------------------------------------------------
# Train all final models
# ------------------------------------------------------------

# Store trained final models
final_models = {}

# Store test probabilities
test_probs_by_family = {
    name: np.zeros((len(X_test), 3))
    for name, _ in model_factories
}

for name, factory in model_factories:

    final_models[name] = []

    for seed in final_seeds:

        print(f"Training {name} seed {seed}")

        model = factory(seed)

        model.fit(
            X_train,
            Y_train
        )

        # Save the trained model
        final_models[name].append(model)

        # Generate test probabilities
        test_probs_by_family[name] += (
            model.predict_proba(X_test)
        )

    # Average predictions from all seeds
    test_probs_by_family[name] /= len(final_seeds)

# ============================================================
# 19. META-LEARNER TEST PREDICTION
# ============================================================

meta_X_test = np.hstack(

    [
        test_probs_by_family[name]

        for name, _
        in model_factories
    ]

)


# Final stacked probabilities

final_prob = (
    meta_model.predict_proba(
        meta_X_test
    )
)


# Final predicted class

Y_pred = np.argmax(
    final_prob,
    axis=1
)


# ============================================================
# 20. METRICS
# ============================================================

accuracy = accuracy_score(

    Y_test,

    Y_pred
)


macro_f1 = f1_score(

    Y_test,

    Y_pred,

    average="macro"
)


weighted_f1 = f1_score(

    Y_test,

    Y_pred,

    average="weighted"
)


balanced_accuracy = balanced_accuracy_score(

    Y_test,

    Y_pred
)


roc_auc = roc_auc_score(

    Y_test,

    final_prob,

    multi_class="ovr",

    average="macro"
)


# ============================================================
# 21. FINAL RESULTS
# ============================================================

print("\n\n==========================================")
print("FINAL MODEL RESULTS (STACKED)")
print("==========================================")


print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    f"Macro F1: {macro_f1:.4f}"
)


print(
    f"Weighted F1: {weighted_f1:.4f}"
)


print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)


print(
    f"ROC AUC: {roc_auc:.4f}"
)


# ============================================================
# 22. CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")


print(

    classification_report(

        Y_test,

        Y_pred,

        target_names=encoder.classes_,

        digits=4

    )

)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")


cm = confusion_matrix(

    Y_test,

    Y_pred

)


print(

    pd.DataFrame(

        cm,

        index=encoder.classes_,

        columns=encoder.classes_

    )

)


# ============================================================
# 24. FEATURE IMPORTANCE
# ============================================================

print("\n==============================")
print("TOP 15 FEATURES")
print("(XGBoost, last seed)")
print("==============================")


xgb_final = create_xgb(
    final_seeds[-1]
)


xgb_final.fit(

    X_train,

    Y_train
)


importances = (

    xgb_final
    .named_steps["model"]
    .feature_importances_

)


importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": importances

})


importance_df = (

    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )

)


print(

    importance_df
    .head(15)
    .to_string(
        index=False
    )

)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")


print(
    "Number of features:",
    len(features)
)


print(
    "Model families used:",
    [
        name
        for name, _
        in model_factories
    ]
)


print(
    "FP flags used:",
    fp_flags
)


print(
    "Error features used:",
    error_features
)


print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    f"Macro F1: {macro_f1:.4f}"
)


print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)


print(
    f"ROC AUC: {roc_auc:.4f}"
)


model_package = {
    "models": final_models,
    "meta_model": meta_model,
    "features": features,
    "label_encoder": encoder,
    "version": "V9",
    "model_families": [
        name for name, _ in model_factories
    ],
    "final_seeds": final_seeds
}

joblib.dump(
    model_package,
    "exo_model_v9.pkl"
)



print("\nComplete  model saved as exo_model_v9.pkl")

print("\nModel saved as exo_model.pkl")

print("\n==========================================")
print("VERSION  FINAL MODEL COMPLETED")
print("==========================================")

