from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "housing.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "housing_clean.csv"

# Global config
RANDOM_STATE = 42
