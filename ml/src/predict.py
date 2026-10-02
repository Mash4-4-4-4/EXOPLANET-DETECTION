import joblib
import numpy as np
import pandas as pd
model_path="exo_model_v7.pkl"

def load_model():
    model=joblib.load(model_path)
    return model

def prepare_input(data, features):

    data = data.copy()

    # 1. Basic engineered features
    data["duration_period_ratio"] = (
        data["koi_duration"] / (data["koi_period"] + 1e-6)
    )

    data["stellar_density_calculated"] = (
        data["koi_smass"] / ((data["koi_srad"] + 1e-6) ** 3)
    )

    data["radius_mass_ratio"] = (
        data["koi_prad"] / (data["koi_smass"] + 1e-6)
    )

    data["planet_star_radius_ratio"] = (
        data["koi_prad"] / (data["koi_srad"] + 1e-6)
    )

    data["snr_per_transit"] = (
        data["koi_model_snr"] / (data["koi_num_transits"] + 1e-6)
    )

    data["depth_ror_interaction"] = (
        data["koi_depth"] * data["koi_ror"]
    )

    data["impact_inclination_interaction"] = (
        data["koi_impact"] * data["koi_incl"]
    )

    data["snr_transit_count"] = (
        data["koi_model_snr"] * data["koi_num_transits"]
    )

    data["depth_ror_ratio"] = (
        data["koi_depth"] / data["koi_ror"].replace(0, np.nan)
    )

    data["semi_major_stellar_radius_ratio"] = (
        data["koi_sma"] / data["koi_srad"].replace(0, np.nan)
    )

    # 2. Error features
    if "koi_period_err1" in data.columns:
        data["period_rel_err"] = (
            abs(data["koi_period_err1"]) /
            (abs(data["koi_period"]) + 1e-6)
        )

    if "koi_prad_err1" in data.columns:
        data["prad_rel_err"] = (
            abs(data["koi_prad_err1"]) /
            (abs(data["koi_prad"]) + 1e-6)
        )

    if "koi_duration_err1" in data.columns:
        data["duration_rel_err"] = (
            abs(data["koi_duration_err1"]) /
            (abs(data["koi_duration"]) + 1e-6)
        )

    if "koi_depth_err1" in data.columns:
        data["depth_rel_err"] = (
            abs(data["koi_depth_err1"]) /
            (abs(data["koi_depth"]) + 1e-6)
        )

    # 3. False-positive flags
    fp_flags = [
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec"
    ]

    for col in fp_flags:
        if col not in data.columns:
            data[col] = 0

    # 4. Keep EXACTLY the features used during training
    data = data[features].copy()

    # 5. Replace infinity with NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    return data


def predict(model_package, data):

    models = model_package["models"]
    meta_model = model_package["meta_model"]

    family_prob = []

    for family in model_package["model_families"]:

        family_models = models[family]

        seed_prob = []

        for model in family_models:

            proba = model.predict_proba(data)

            seed_prob.append(proba)

        family_prob.append(
            np.mean(seed_prob, axis=0)
        )

    meta_input = np.hstack(family_prob)

    final_proba = meta_model.predict_proba(meta_input)

    predicted_class = np.argmax(
        final_proba,
        axis=1
    )

    label_encoder = model_package["label_encoder"]

    prediction = label_encoder.inverse_transform(
        predicted_class
    )

    return prediction, final_proba

def main():
        
    model_package=load_model()
    data=pd.read_csv("input.csv")
    data=prepare_input(data,model_package["features"])
    prediction,probability=predict(model_package,data)
    
    print("Prediction:", prediction)
    print("Probability:", probability)
    
if __name__=="__main__":
    main()
        
    
    