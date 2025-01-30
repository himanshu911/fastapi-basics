from fastai.tabular.all import *
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


# Step 1: Load the California Housing Dataset
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df


# Step 2: Preprocess the Data
def preprocess_data(df):
    # Rename target column for clarity
    df = df.rename(columns={"MedHouseVal": "MedianHouseValue"})

    # Feature Engineering (if needed)
    # For simplicity, we'll use the existing features

    return df


# Step 3: Create FastAI DataLoaders
def create_dataloaders(df):
    # Define categorical and continuous variables
    # In this dataset, all features are continuous
    cont_names = list(df.columns)
    cont_names.remove("MedianHouseValue")
    cat_names = []  # No categorical variables in this dataset

    # Define the procs (preprocessors)
    procs = [Normalize()]

    # Split the data into training and validation sets
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create DataLoaders
    dls = TabularDataLoaders.from_df(
        train_df,
        path=Path("."),
        valid_df=valid_df,
        y_names="MedianHouseValue",
        cat_names=cat_names,
        cont_names=cont_names,
        procs=procs,
        y_block=RegressionBlock(),
        bs=64,
    )

    return dls


# Step 4: Define and Train the FCNN Model
def train_model(dls):
    # Define the learner with a simple FCNN architecture
    learn = tabular_learner(
        dls, layers=[200, 100], metrics=mae, loss_func=MSELossFlat()
    )

    # Train the model
    learn.fit_one_cycle(10, 1e-2)

    return learn


# Step 5: Save the Trained Model
def save_model(learn):
    # Ensure the 'app' directory exists
    Path("app").mkdir(parents=True, exist_ok=True)

    # Export the model for inference
    learn.export("app/housing_price_model.pkl")
    print("Model trained and saved successfully.")


def main():
    df = load_data()
    df = preprocess_data(df)
    dls = create_dataloaders(df)
    learn = train_model(dls)
    save_model(learn)


if __name__ == "__main__":
    main()
