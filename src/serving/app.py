from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

app = FastAPI(title="Self-Healing ML Inference API")

# 🔹 Load production model from MLflow
MODEL_NAME = "housing_price_model"
MODEL_STAGE = "Production"

import mlflow.pyfunc
import os

MODEL_PATH = "/app/mlruns/0/models"

latest_model = sorted(os.listdir(MODEL_PATH))[-1]

model = mlflow.pyfunc.load_model(
    f"{MODEL_PATH}/{latest_model}/artifacts"
)


# 🔹 Input schema
class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity_INLAND: float


@app.get("/health")
def health():
    return {"status": "Model is live 🚀"}


@app.post("/predict")
def predict(data: HousingInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}
from pydantic import BaseModel
import pandas as pd

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    preds = model.predict(df)
    return {"prediction": preds.tolist()}
