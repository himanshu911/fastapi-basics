import pytest
import pandas as pd
from app.services import predict_service
from pydantic import BaseModel


# Mock input features
class MockFeatures(BaseModel):
    MedInc: float = 8.3252
    HouseAge: float = 41.0
    AveRooms: float = 6.984127
    AveBedrms: float = 1.023810
    Population: float = 322.0
    AveOccup: float = 2.555556
    Latitude: float = 37.88
    Longitude: float = -122.23


# Mock model
class MockModel:
    def predict(self, input_data):
        return [pd.DataFrame({"MedianHouseValue": [300000.0]})]

    @property
    def dls(self):
        class DLS:
            y_names = ["MedianHouseValue"]

        return DLS()


# Test the predict_service function
def test_predict_service():
    features = MockFeatures()
    model = MockModel()
    predicted_price = predict_service(features, model)
    assert predicted_price == 300000.0
