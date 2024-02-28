import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
# from db_Feature import db_Features
import numpy as np
import pickle

# Define Pydantic model for input data validation
class InputFeatures(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Create the FastAPI app object
app = FastAPI()

# Load the trained model
with open('Classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)
    
# Define the root route
@app.get('/')
def home():
    return {'Hello': 'Welome to my page'}

# Define the root route
@app.get('/{name}')
def give_name(name: str):
    return ('Hello '+ name +  '.Welcome to DiabPredictor!!,   The best predictor of diabetes status of your patient')
 
 
# Define the prediction endpoint
@app.post('/predict')
def predict_status(data: InputFeatures):
    # Extract data from the input
    pregnancies = data.Pregnancies
    Glucose = data.Glucose
    BloodPressure = data.BloodPressure
    SkinThickness = data.SkinThickness
    Insulin = data.Insulin
    BMI = data.BMI
    DiabetesPedigreeFunction = data.DiabetesPedigreeFunction
    Age = data.Age
    
    # Make prediction
    prediction = classifier.predict([[pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    if prediction[0]=='yes':
        message='The person has diabetes'
    else:
        message='The person has no diabetes'   
        
    return {'prediction': prediction[0], 'Message':message}

