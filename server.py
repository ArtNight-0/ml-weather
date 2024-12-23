from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained models
economic_model = joblib.load('economic_growth_model.pkl')
birth_model = joblib.load('arima_birth_rate_model.pkl')

# Create a FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Economic Growth Prediction Models
class EconomicData(BaseModel):
    interest_rate: float
    unemployment_rate: float

@app.post("/predict/economic-growth")
def predict_economic_growth(data: EconomicData):
    # Convert input data to the format expected by the model
    input_data = np.array([[data.interest_rate, data.unemployment_rate]])
    
    # Make a prediction
    prediction = economic_model.predict(input_data)[0]
    
    # Return the prediction as a JSON response
    return {"predicted_gdp_growth_rate": prediction}

# Birth Rate Prediction Models
class PredictionInput(BaseModel):
    steps: int = Field(..., gt=0, le=120)  # Number of future steps to predict, with validation

@app.post("/predict/birth-rate")
def predict_birth_rate(data: PredictionInput):
    # Get the number of steps to predict
    steps = data.steps

    logging.info(f"Received steps: {steps}")

    try:
        # Use the model to forecast
        forecast = birth_model.forecast(steps=steps)
        logging.info(f"Forecast result: {forecast}")
    except pd.errors.OutOfBoundsDatetime as e:
        raise HTTPException(status_code=400, detail="Number of steps is too large, resulting in out-of-bounds datetime.")

    # Return the forecast as a list
    return {"predicted_birth_rates": forecast.tolist()}

# To run the server: uvicorn server:app --reload
