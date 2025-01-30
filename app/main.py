import logging
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from .services import load_model, predict_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


# Define the input data model
class HousingFeatures(BaseModel):
    MedInc: float = 3.0  # Median income in block group
    HouseAge: float = 30.0  # Median house age in block group
    AveRooms: float  # Average number of rooms per household
    AveBedrms: float  # Average number of bedrooms per household
    Population: float  # Population of block group
    AveOccup: float  # Average number of occupants per household
    Latitude: float  # Latitude coordinate
    Longitude: float  # Longitude coordinate


# Define the response model
class PredictionResult(BaseModel):
    predicted_price: float  # Predicted house price


@app.get("/")
def home():
    return {"message": "Welcome to the Housing Price Prediction API!"}


@app.post("/predict", response_model=PredictionResult)
def predict_price(features: HousingFeatures, model=Depends(load_model)):
    logger.info(f"Received prediction request: {features}")
    try:
        predicted_price = predict_service(features, model)
        return PredictionResult(predicted_price=predicted_price)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy"}
