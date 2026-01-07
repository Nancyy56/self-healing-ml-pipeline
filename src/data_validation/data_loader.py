import pandas as pd
from config.constants import RAW_DATA_PATH

DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    "datasets/housing/housing.csv"
)

def load_and_save_raw_data():
    df = pd.read_csv(DATA_URL)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Raw data saved at: {RAW_DATA_PATH}")
    return df


if __name__ == "__main__":
    load_and_save_raw_data()

