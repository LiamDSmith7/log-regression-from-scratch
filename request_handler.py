from fastapi import FastAPI
import numpy as np

from src.model import LogisticRegressionScratch

app = FastAPI()

model = LogisticRegressionScratch()

# In a real version, load these from saved files
model.weights = np.array([...])
model.bias = 0.5
mean = np.array([...])
std = np.array([...])

@app.post("/predict")
def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)

    X_scaled = (X - mean) / std

    probability = model.predict_proba(X_scaled)[0]
    prediction = model.predict(X_scaled)[0]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }