from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('urbanization_growth_model.pkl')

# Create a FastAPI app
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class EconomicData(BaseModel):
    gdp_per_capita: float

# Define a route for prediction
@app.post("/predict")
def predict_economic_growth(data: EconomicData):
    # Convert input data to the format expected by the model
    input_data = np.array([data.gdp_per_capita])
    
    # Make a prediction
    prediction = model.predict(input_data.reshape(-1, 1))[0]
    
    # Return the prediction as a JSON response
    return {"urbanization_rate": prediction}

# To run the server: uvicorn main:app --reload
