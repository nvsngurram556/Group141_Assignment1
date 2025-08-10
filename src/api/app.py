from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import numpy as np
import os
import logging
import sqlite3
import subprocess


# Configure logging
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("app.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

app = FastAPI()
# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency in seconds", ["method", "endpoint"]
)


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    return response

@app.get("/metrics")
async def metrics():
    """
    Endpoint to expose Prometheus metrics.
    """
    return generate_latest(), {"Content-Type": CONTENT_TYPE_LATEST}


@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the California Housing Prediction API"}

    
conn = sqlite3.connect("predictions.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    client_ip TEXT,
    input_json TEXT,
    prediction REAL
)
""")
conn.commit()
# Initialize Prometheus instrumentation
Instrumentator = Instrumentator()
Instrumentator.instrument(app).expose(app)


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
async def predict(data: HousingData, request: Request):
    try:
        # Keep consistent order of features
        features = np.array([[data.MedInc, data.HouseAge, data.AveRooms,
                              data.AveBedrms, data.Population, data.AveOccup,
                              data.Latitude, data.Longitude, data.MedHouseVal]])
        prediction = model.predict(features)
        c.execute(
            "INSERT INTO prediction_logs (timestamp, client_ip, input_json, prediction) "
            "VALUES (datetime('now'), ?, ?, ?)",
            (
                request.client.host,
                str(data.dict()),
                float(prediction[0])
            )
        )
        conn.commit()
        logger.info(f"Prediction request from {request.client.host}: {features}")
        return {"prediction": float(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction error from {request.client.host}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain():
    try:
        result = subprocess.run(
            ["python", "src/models/train_model.py", "--experiment-name", "california-housing"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        # Reload the model (assuming you save the best model to a known path)
        global model
        model = joblib.load("mlruns/518425936383115905/34a7276f614c4a7aa1d18fc6fb047ad9/artifacts/model/model.pkl")
        logger.info("Model retrained and reloaded successfully.")
        return {"status": "Model retrained successfully"}
    except subprocess.CalledProcessError as e:
        logger.error(f"Retrain failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e.stderr}")
