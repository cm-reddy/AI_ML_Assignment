from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
# import joblib  # If using local .pkl file

# Load MLflow model
model_uri = "runs:/70968cdab4644053835a226c51eec164/model"
model = mlflow.pyfunc.load_model(model_uri)

# OR load local model:
# model = joblib.load("models/RandomForest.pkl")

# Feature names expected by the MLflow model
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# FastAPI setup
app = FastAPI()

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris prediction API is running."}

@app.post("/predict")
def predict(data: IrisRequest):
    input_df = pd.DataFrame([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], columns=FEATURE_NAMES)

    prediction = model.predict(input_df)

    return {"predicted_class": int(prediction[0])}
