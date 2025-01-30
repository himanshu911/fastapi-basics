import pandas as pd
from fastai.tabular.all import load_learner
from fastapi import HTTPException


# Load the model
def load_model():
    try:
        model = load_learner("app/housing_price_model.pkl")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model.")


# Prediction service
def predict_service(features, model):
    try:
        input_data = pd.Series(features.dict())
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0].loc[0, model.dls.y_names[0]])
        return predicted_price
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
