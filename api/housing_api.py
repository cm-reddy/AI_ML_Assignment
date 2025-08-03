from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

app = FastAPI()

model_uri = "runs:/4bf65a1a6fdd4d9fb80d35b460d5d721/model"
model = mlflow.pyfunc.load_model(model_uri)

class HousingRequest(BaseModel):
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    housing_median_age: float
    latitude: float
    longitude: float

@app.get("/")
def root():
    return {"message": "Housing price prediction API is running."}

@app.post("/predict")
def predict(data: HousingRequest):
    # Create DataFrame
    df = pd.DataFrame([data.dict()])

    # Feature engineering (must match training script!)
    df["AveRooms"] = df["total_rooms"] / df["households"]
    df["AveBedrms"] = df["total_bedrooms"] / df["households"]
    df["AveOccup"] = df["population"] / df["households"]

    # Rename to match training column names
    df.rename(columns={
        "median_income": "MedInc",
        "housing_median_age": "HouseAge",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "population": "Population"
    }, inplace=True)

    final_features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]

    prediction = model.predict(df[final_features])
    return {"predicted_price": float(prediction[0])}
