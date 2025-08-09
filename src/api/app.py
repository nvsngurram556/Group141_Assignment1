from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Input schema
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    MedHouseVal: float
# Load the model
model_path = "mlruns/518425936383115905/34a7276f614c4a7aa1d18fc6fb047ad9/artifacts/model/model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

@app.post("/predict")
def predict(data: HousingData):
    try:
        # Keep consistent order of features
        features = np.array([[data.MedInc, data.HouseAge, data.AveRooms,
                              data.AveBedrms, data.Population, data.AveOccup,
                              data.Latitude, data.Longitude, data.MedHouseVal]])
        prediction = model.predict(features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))