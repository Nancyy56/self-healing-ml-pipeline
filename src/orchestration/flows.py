from prefect import flow, task
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from config.constants import PROCESSED_DATA_PATH, RANDOM_STATE
from src.training.feature_engineering import prepare_features
from src.training.model_evaluation import evaluate_model

from src.data_validation.drift_detector import generate_drift_report, DRIFT_THRESHOLD
from prefect import get_run_logger
from src.model_registry.model_promoter import promote_model_if_better


TARGET_COLUMN = "median_house_value"


@task(retries=2, retry_delay_seconds=10)
def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


@task
def prepare_data(df):
    return prepare_features(df, TARGET_COLUMN)


@task
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


@task
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    return evaluate_model(y_test, predictions)


@task
def log_to_mlflow(model, metrics):
    import mlflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="housing_price_model")
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        run_id = run.info.run_id
    return run_id  # return this run_id for promotion

@task
def check_drift():
    """
    Runs drift detection and returns True if retraining is needed.
    """
    logger = get_run_logger()
    overall_drift = generate_drift_report()  # returns 0.0 to 1.0
    if overall_drift > DRIFT_THRESHOLD:
        logger.info(f"⚠️ Drift detected! Score = {overall_drift:.2f} > threshold {DRIFT_THRESHOLD}")
        return True  # trigger retraining
    else:
        logger.info(f"✅ Drift low. Score = {overall_drift:.2f} <= threshold {DRIFT_THRESHOLD}")
        return False  # skip retraining


@flow(name="self_healing_training_pipeline")
def training_flow():
    # 1️⃣ Check drift first
    retrain_needed = check_drift()

    # 2️⃣ Only retrain if drift is above threshold
    if retrain_needed:
        df = load_data()
        X, y = prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        model = train_model(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        # Log model and get run_id
        run_id = log_to_mlflow(model, metrics)

        # Promote model if better
        from src.model_registry.model_promoter import promote_model_if_better
        promote_model_if_better(run_id)

    else:
        print("Skipping retraining — drift below threshold.")



if __name__ == "__main__":
    training_flow()
