import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "housing_price_model"
METRIC_NAME = "rmse"   # lower is better


def promote_model_if_better(new_run_id: str):
    """
    Compare new model with production model.
    Promote only if new model is better.
    """

    client = MlflowClient()

    # Get current production model (if exists)
    try:
        prod_versions = client.get_latest_versions(
            MODEL_NAME, stages=["Production"]
        )
        prod_run_id = prod_versions[0].run_id
        prod_rmse = client.get_run(prod_run_id).data.metrics[METRIC_NAME]
    except Exception:
        print("No production model found. Promoting new model directly.")
        prod_rmse = float("inf")

    # Get new model metric
    new_rmse = client.get_run(new_run_id).data.metrics[METRIC_NAME]

    print(f"Production RMSE: {prod_rmse}")
    print(f"New Model RMSE: {new_rmse}")

    if new_rmse < prod_rmse:
        print("✅ New model is better. Promoting to Production.")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=get_model_version(new_run_id, client),
            stage="Production",
            archive_existing_versions=True
        )
    else:
        print("❌ New model is worse. Keeping existing Production model.")


def get_model_version(run_id, client):
    versions = client.search_model_versions(f"run_id='{run_id}'")
    return versions[0].version

