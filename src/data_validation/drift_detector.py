import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os

REFERENCE_DATA_PATH = "data/processed/reference_data.csv"
CURRENT_DATA_PATH = "data/processed/current_data.csv"

DRIFT_THRESHOLD = 0.3


def generate_reference_data():
    if not os.path.exists(REFERENCE_DATA_PATH):
        df = pd.read_csv("data/processed/housing_clean.csv")
        df.to_csv(REFERENCE_DATA_PATH, index=False)


def generate_drift_report():
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    current = pd.read_csv(CURRENT_DATA_PATH)

    drifted_features = 0
    total_features = 0

    for col in reference.columns:
        if reference[col].dtype != "object":
            stat, p_value = ks_2samp(reference[col], current[col])
            total_features += 1
            if p_value < 0.05:
                drifted_features += 1

    drift_score = drifted_features / total_features

    print(f"Detected drift in {drifted_features}/{total_features} features")
    print(f"Overall drift score: {drift_score:.2f}")

    return drift_score

