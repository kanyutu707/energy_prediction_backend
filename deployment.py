import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import uvicorn

from fastapi.middleware.cors import CORSMiddleware


with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

app = FastAPI(title="Energy Consumption Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

day_encoder = LabelEncoder()
holiday_encoder = LabelEncoder()
hvac_encoder = LabelEncoder()
lighting_encoder = LabelEncoder()


class ModelInput(BaseModel):
    Month: int
    Hour: int 
    DayOfWeek: str  
    Holiday: str   
    Temperature: float
    Humidity: float
    SquareFootage: float
    Occupancy: int
    HVACUsage: str
    LightingUsage: str
    RenewableEnergy: float

    class Config:
        schema_extra = {
            "example": {
                "Month": 2,
                "Hour": 4,
                "DayOfWeek": "Monday",
                "Holiday": "No",
                "Temperature": 24.32,
                "Humidity": 42.51,
                "SquareFootage": 1413.22,
                "Occupancy": 3,
                "HVACUsage": "Off",
                "LightingUsage": "On",
                "RenewableEnergy": 5.32
            }
        }

class PredictionResponse(BaseModel):
    predicted_energy_usage: float

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: ModelInput):
  
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    categorical_cols = ['DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage']
   
    try:
        for col in categorical_cols:
            if col == 'DayOfWeek':
                input_df[col] = day_encoder.fit_transform(input_df[col])
            elif col == 'Holiday':
                input_df[col] = holiday_encoder.fit_transform(input_df[col])
            elif col == 'HVACUsage':
                input_df[col] = hvac_encoder.fit_transform(input_df[col])
            elif col == 'LightingUsage':
                input_df[col] = lighting_encoder.fit_transform(input_df[col])
    except Exception as e:
        return {"error": f"Error encoding categorical features: {str(e)}"}
    
    try:
        prediction = loaded_model.predict(input_df)
        return {"predicted_energy_usage": float(prediction[0])}
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.get("/")
def read_root():
    return {"message": "Energy Consumption Prediction API", 
            "usage": "Send POST request to /predict endpoint with required parameters"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9030)