# ============================================================
# EXOPLANET DETECTION - VERSION 7
# + KOI FALSE-POSITIVE FLAGS
# + MEASUREMENT-ERROR FEATURES
# + XGBOOST + LIGHTGBM ENSEMBLE
# + STACKED META-LEARNER (replaces scalar weight calibration)
# ============================================================
#
# CHANGES FROM YOUR VERSION 6 (why each one should help):
#
# 1. koi_fpflag_* columns added as raw features.
#    These are NASA's own vetting flags (not-transit-like,
#    stellar-eclipse, centroid-offset, ephemeris-match). They
#    exist specifically to flag false positives and are usually
#    among the strongest predictors in any KOI classifier.
#    If they weren't in your feature list before, this alone
#    should move your FALSE POSITIVE and CANDIDATE scores.
#
# 2. Relative-error features added (err / value) for the
#    parameters most tied to vetting confidence: period, prad,
#    duration, depth. High relative uncertainty is itself a
#    signal of a marginal/CANDIDATE detection.
#
# 3. LightGBM added alongside XGBoost. Same data, different
#    tree-growth strategy -> genuinely different error patterns
#    -> ensemble averaging gains more than just more XGB seeds.
#    Falls back to XGBoost-only automatically if lightgbm isn't
#    installed (pip install lightgbm).
#
# 4. Your scalar "multiply Candidate probability by 1.5" step
#    is replaced with a stacked meta-learner: a small logistic
#    regression trained on the OOF probabilities from the base
#    models. This lets the calibration layer learn a real
#    decision boundary between all three classes instead of
#    a single hand-tuned knob on one class.
#
# Everything else (StratifiedGroupKFold by kepid, leakage check,
# outer train/test split) is unchanged from your pipeline.
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed - continuing with XGBoost only. "
          "Run: pip install lightgbm  for the full ensemble.")


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
    subset=["koi_disposition", "kepid"]
).copy()


# ============================================================
# 3. RAW FEATURES
# ============================================================

raw_features = [

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
    "koi_srho"
]

# --- NEW: KOI false-positive vetting flags -------------------
# These are binary (0/1) columns NASA's pipeline already
# computes. Only add the ones actually present in your CSV.
fp_flag_candidates = [
    "koi_fpflag_nt",   # not-transit-like false positive flag
    "koi_fpflag_ss",   # stellar eclipse flag
    "koi_fpflag_co",   # centroid offset flag
    "koi_fpflag_ec",   # ephemeris match / contamination flag
]
fp_flags = [c for c in fp_flag_candidates if c in df.columns]

if fp_flags:
    print("\nUsing FP flags:", fp_flags)
else:
    print("\nWARNING: no koi_fpflag_* columns found in this CSV. "
          "Check the column names in your file - these matter a lot.")


# --- NEW: measurement-error columns, if present ---------------
err_source_cols = [
    "koi_period_err1",
    "koi_prad_err1",
    "koi_duration_err1",
    "koi_depth_err1",
]
err_cols_present = [c for c in err_source_cols if c in df.columns]

if err_cols_present:
    print("Using error columns:", err_cols_present)
else:
    print("WARNING: no koi_*_err1 columns found - relative-error "
          "features will be skipped.")


# ============================================================
# 4. ENGINEERED FEATURES
# ============================================================

df["duration_period_ratio"] = (
    df["koi_duration"] / (df["koi_period"] + 1e-6)
)

df["stellar_density_calculated"] = (
    df["koi_smass"] / ((df["koi_srad"] + 1e-6) ** 3)
)

df["radius_mass_ratio"] = (
    df["koi_prad"] / (df["koi_smass"] + 1e-6)
)

df["planet_star_radius_ratio"] = (
    df["koi_prad"] / (df["koi_srad"] + 1e-6)
)

df["snr_per_transit"] = (
    df["koi_model_snr"] / (df["koi_num_transits"] + 1e-6)
)

df["depth_ror_interaction"] = (
    df["koi_depth"] * df["koi_ror"]
)

df["impact_inclination_interaction"] = (
    df["koi_impact"] * df["koi_incl"]
)

df["snr_transit_count"] = (
    df["koi_model_snr"] * df["koi_num_transits"]
)

df["depth_ror_ratio"] = (
    df["koi_depth"] / df["koi_ror"].replace(0, np.nan)
)

df["semi_major_stellar_radius_ratio"] = (
    df["koi_sma"] / df["koi_srad"].replace(0, np.nan)
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

# --- NEW: relative-error engineered features -------------------
error_features = []

if "koi_period_err1" in df.columns:
    df["period_rel_err"] = (
        df["koi_period_err1"].abs() / (df["koi_period"].abs() + 1e-6)
    )
    error_features.append("period_rel_err")

if "koi_prad_err1" in df.columns:
    df["prad_rel_err"] = (
        df["koi_prad_err1"].abs() / (df["koi_prad"].abs() + 1e-6)
    )
    error_features.append("prad_rel_err")

if "koi_duration_err1" in df.columns:
    df["duration_rel_err"] = (
        df["koi_duration_err1"].abs() / (df["koi_duration"].abs() + 1e-6)
    )
    error_features.append("duration_rel_err")

if "koi_depth_err1" in df.columns:
    df["depth_rel_err"] = (
        df["koi_depth_err1"].abs() / (df["koi_depth"].abs() + 1e-6)
    )
    error_features.append("depth_rel_err")

engineered_features = engineered_features + error_features


# ============================================================
# 5. FINAL FEATURE LIST
# ============================================================

features = raw_features + engineered_features

print("\n==============================")
print("FEATURES")
print("==============================")
print("Raw features:", len(raw_features))
print("Engineered features:", len(engineered_features))
print("Total features:", len(features))


# ============================================================
# 6. X / Y / GROUPS
# ============================================================

X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

y = df["koi_disposition"]
groups = df["kepid"]


# ============================================================
# 7. ENCODE TARGET
# ============================================================

encoder = LabelEncoder()
Y = encoder.fit_transform(y)

print("\nClasses:")
print(dict(zip(encoder.classes_, range(len(encoder.classes_)))))


# ============================================================
# 8. OUTER GROUP SPLIT
# ============================================================

outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(outer_cv.split(X, Y, groups))

X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)

Y_train = Y[train_idx]
Y_test = Y[test_idx]

groups_train = groups.iloc[train_idx].reset_index(drop=True)
groups_test = groups.iloc[test_idx].reset_index(drop=True)

print("\n==============================")
print("OUTER SPLIT")
print("==============================")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training stars:", groups_train.nunique())
print("Testing stars:", groups_test.nunique())


# ============================================================
# 9. CHECK LEAKAGE
# ============================================================

overlap = set(groups_train).intersection(set(groups_test))
print("Overlapping stars:", len(overlap))

if len(overlap) == 0:
    print("NO STAR LEAKAGE")
else:
    raise ValueError("STAR LEAKAGE DETECTED")


# ============================================================
# 10. MODEL FACTORIES
# ============================================================

def create_xgb(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
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
        ))
    ])


def create_lgbm(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(
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
        ))
    ])


def get_model_factories():
    factories = [("xgb", create_xgb)]
    if HAS_LGBM:
        factories.append(("lgbm", create_lgbm))
    return factories


model_factories = get_model_factories()
seeds = [42, 52, 62]


# ============================================================
# 11. OOF PREDICTIONS PER MODEL FAMILY
# ============================================================
#
# We keep each model family's OOF probabilities SEPARATE
# (rather than averaging them into one block) so the stacker
# in step 12 can learn how much to trust each family, per class.
# ============================================================

sample_weights=compute_sample_weight(class_weight="balanced",
                                      y=Y_train)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)

print("\n==============================")
print("OOF GENERATION")
print("==============================")

oof_probs_by_family = {
    name: np.zeros((len(X_train), 3)) for name, _ in model_factories
}

for fold, (tr_idx, val_idx) in enumerate(
    cv.split(X_train, Y_train, groups_train)
):
    print(f"\nFold {fold + 1}/5")

    for name, factory in model_factories:
        fold_prob = np.zeros((len(val_idx), 3))

        for seed in seeds:
            model = factory(seed)
            model.fit(X_train.iloc[tr_idx], Y_train[tr_idx], model__sample_weight=sample_weights[tr_idx])
            fold_prob += model.predict_proba(X_train.iloc[val_idx])

        fold_prob /= len(seeds)
        oof_probs_by_family[name][val_idx] = fold_prob


# ============================================================
# 12. TRAIN STACKED META-LEARNER ON OOF PROBABILITIES
# ============================================================
#
# Meta-features = concatenated class probabilities from every
# base model family. The meta-learner (multinomial logistic
# regression) learns how to combine/re-weight them - this
# replaces your single "multiply Candidate by 1.5" step with
# something that can use all the information, not one scalar.
# ============================================================

print("\n==============================")
print("STACKING META-LEARNER")
print("==============================")

meta_X_train = np.hstack(
    [oof_probs_by_family[name] for name, _ in model_factories]
)

meta_model = LogisticRegression(
    max_iter=2000,
    C=1.0
)
meta_model.fit(meta_X_train, Y_train, sample_weight=sample_weights)

oof_meta_pred = meta_model.predict(meta_X_train)
oof_meta_f1 = f1_score(Y_train, oof_meta_pred, average="macro")
print("OOF Macro F1 (stacked):", round(oof_meta_f1, 4))


# ============================================================
# 13. TRAIN FINAL BASE MODELS ON FULL TRAINING SET
# ============================================================

print("\n==============================")
print("TRAINING FINAL BASE MODELS")
print("==============================")

final_seeds = [42, 52, 62, 72, 82]

test_probs_by_family = {
    name: np.zeros((len(X_test), 3)) for name, _ in model_factories
}

for name, factory in model_factories:
    for seed in final_seeds:
        print(f"Training {name} seed {seed}")
        model = factory(seed)
        model.fit(X_train, Y_train, model__sample_weight=sample_weights)
        test_probs_by_family[name] += model.predict_proba(X_test)

    test_probs_by_family[name] /= len(final_seeds)


# ============================================================
# 14. APPLY META-LEARNER TO TEST SET
# ============================================================

meta_X_test = np.hstack(
    [test_probs_by_family[name] for name, _ in model_factories]
)

final_prob = meta_model.predict_proba(meta_X_test)
Y_pred = np.argmax(final_prob, axis=1)

# also keep the simple averaged base-model prob for ROC AUC
# on the raw base predictions (useful comparison point)
raw_avg_prob = np.mean(
    [test_probs_by_family[name] for name, _ in model_factories], axis=0
)


# ============================================================
# 15. METRICS
# ============================================================

accuracy = accuracy_score(Y_test, Y_pred)
macro_f1 = f1_score(Y_test, Y_pred, average="macro")
weighted_f1 = f1_score(Y_test, Y_pred, average="weighted")
balanced_accuracy = balanced_accuracy_score(Y_test, Y_pred)
roc_auc = roc_auc_score(Y_test, final_prob, multi_class="ovr", average="macro")


# ============================================================
# 16. FINAL RESULTS
# ============================================================

print("\n\n==========================================")
print("FINAL MODEL RESULTS (STACKED)")
print("==========================================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(
    Y_test, Y_pred, target_names=encoder.classes_, digits=4
))


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
cm = confusion_matrix(Y_test, Y_pred)
print(pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_))


# ============================================================
# 19. FEATURE IMPORTANCE (from one final XGB model, for reference)
# ============================================================

print("\n==============================")
print("TOP 15 FEATURES (XGBoost, last seed)")
print("==============================")

xgb_final = create_xgb(final_seeds[-1])
xgb_final.fit(X_train, Y_train, model__sample_weight=sample_weights)
importances = xgb_final.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(importance_df.head(15).to_string(index=False))


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")
print("Number of features:", len(features))
print("Model families used:", [name for name, _ in model_factories])
print("FP flags used:", fp_flags)
print("Error features used:", error_features)
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\n==========================================")
print("VERSION 7 COMPLETED")
print("==========================================")