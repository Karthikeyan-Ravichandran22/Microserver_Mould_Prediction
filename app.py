# # -*- coding: utf-8 -*-
# """
# Created on Tue Nov 17 21:40:41 2020
# @author: win10
# """

# # Library imports
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, parse_obj_as
# import pandas as pd
# import joblib
# import os

# # Create the app object
# app = FastAPI()

# # Load the trained model, scaler, and schema
# model_path = 'trained_random_forest_classifier.pkl'
# scaler_path = 'trained_scaler.pkl'
# schema_path = 'data_schema.pkl'

# model = joblib.load(model_path)
# scaler = joblib.load(scaler_path)
# schema = joblib.load(schema_path)

# # Function to prepare and make predictions
# def prepare_and_predict(input_df, model, scaler, schema):
#     # Prepare input DataFrame for prediction
#     input_df_encoded = pd.get_dummies(input_df)
#     missing_cols = set(schema) - set(input_df_encoded.columns)
#     for c in missing_cols:
#         input_df_encoded[c] = 0
#     input_df_aligned = input_df_encoded.reindex(columns=schema, fill_value=0)

#     # Scale the features and make predictions
#     input_df_scaled = scaler.transform(input_df_aligned)
#     predictions = model.predict(input_df_scaled)
#     input_df['Predictions'] = predictions
#     return input_df

# # Define a Pydantic model for the DataFrame structure
# class DataFrameInput(BaseModel):
#     data: list
#     columns: list

# @app.post('/predict_dataframe/')
# def predict_dataframe(input: DataFrameInput):
#     try:
#         # Convert the input to a DataFrame
#         input_df = pd.DataFrame(input.data, columns=input.columns)

#         # Make predictions and get output DataFrame
#         output_df = prepare_and_predict(input_df, model, scaler, schema)
#         return output_df.to_dict(orient='records')
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Index route
# @app.get('/')
# def index():
#     return {'message': 'Welcome to the prediction service'}

# # Run the API with uvicorn
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

# -----    with csv 

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020
@author: win10
"""

# Library imports
import uvicorn
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

# Create the app object
app = FastAPI()

# Load the trained model, scaler, and schema
model_path = 'trained_random_forest_classifier.pkl'
scaler_path = 'trained_scaler.pkl'
schema_path = 'data_schema.pkl'


output =pd.read_csv("output.csv")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
schema = joblib.load(schema_path)

# Assume 'output' DataFrame is already defined and accessible here
# Example: output = pd.DataFrame(...)

# Function to prepare and make predictions
def prepare_and_predict(input_df, model, scaler, schema):
    input_df_encoded = pd.get_dummies(input_df)
    missing_cols = set(schema) - set(input_df_encoded.columns)
    for c in missing_cols:
        input_df_encoded[c] = 0
    input_df_aligned = input_df_encoded.reindex(columns=schema, fill_value=0)
    input_df_scaled = scaler.transform(input_df_aligned)
    predictions = model.predict(input_df_scaled)
    input_df['Predictions'] = predictions
    return input_df

@app.post('/use_existing_dataframe/')
def use_existing_dataframe():
    try:
        # Use the existing 'output' DataFrame for predictions
        global output
        output_with_predictions = prepare_and_predict(output, model, scaler, schema)

        # Save to CSV
        output_file_path = 'predicted_output.csv'
        output_with_predictions.to_csv(output_file_path, index=False)

        return {"message": "Predictions made and saved to CSV", "file_path": output_file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Index route
@app.get('/')
def index():
    return {'message': 'Welcome to the prediction service'}

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#  Run the post API
# curl -X POST http://127.0.0.1:8000/use_existing_dataframe/
