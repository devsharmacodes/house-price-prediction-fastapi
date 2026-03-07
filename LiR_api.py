from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Load trained model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Request schema
class HouseFeatures(BaseModel):
    GrLivArea: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict_price(data: HouseFeatures):

    # Convert input to array
    features = np.array([[data.GrLivArea]])

    # Prediction
    prediction = model.predict(features)

    return {
        "Predicted House Price": float(prediction[0])
    }