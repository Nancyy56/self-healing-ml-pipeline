import pandas as pd
from config.constants import RAW_DATA_PATH, PROCESSED_DATA_PATH


def clean_data():
    df = pd.read_csv(RAW_DATA_PATH)

    df = df.drop_duplicates()
    df = df.dropna()

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Clean data saved at: {PROCESSED_DATA_PATH}")
    return df


if __name__ == "__main__":
    clean_data()
