# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os

# Load the trained model and scaler

# 2. Create the app object
app = FastAPI()
import pickle

# Update these paths to the absolute paths of your pickle files
model_path = 'trained_random_forest_classifier.pkl'
scaler_path = 'trained_scaler.pkl'

# Test loading the model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")

# Test loading the scaler
try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Failed to load scaler: {e}")


# Define a Pydantic model for the input data structure
class PredictionInput(BaseModel):
    width: float
    length: float
    floor_area: float
    window_width: float
    window_height: float
    provided_purge: float
    required_purge: float

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, welcome to the prediction service'}

# 4. Prediction endpoint
@app.post('/predict')
def make_prediction(input_data: PredictionInput):
    try:
        input_df = pd.DataFrame([input_data.dict()])
        input_df_scaled = scaler.transform(input_df)
        prediction = classifier.predict(input_df_scaled)
        return {"prediction": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
