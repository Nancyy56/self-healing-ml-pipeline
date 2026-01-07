import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from config.constants import PROCESSED_DATA_PATH, RANDOM_STATE
from src.training.feature_engineering import prepare_features
from src.training.model_evaluation import evaluate_model


TARGET_COLUMN = "median_house_value"

def train():
    # Load data
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Split features and target
    X, y = prepare_features(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("model_type", "RandomForestRegressor")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path ="model", registered_model_name = "housing_price_model")

        print("Training complete")
        print(metrics)


if __name__ == "__main__":
    train()
