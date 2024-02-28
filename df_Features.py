from pydantic import BaseModel

# 2. Lets create a class for the the data we may need for the prediction
class df_Features(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    