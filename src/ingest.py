import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing

def ingest_data():
    os.makedirs("data", exist_ok=True)
    
    # Disease Dataset (Classification)
    disease_data = load_breast_cancer(as_frame=True)
    df_disease = disease_data.frame
    df_disease.to_csv("data/disease_dataset.csv", index=False)
    print("Disease dataset saved to data/disease_dataset.csv")

    # Housing Dataset
    housing_data = fetch_california_housing(as_frame=True)
    df_housing = housing_data.frame
    df_housing.to_csv("data/boston_housing.csv", index=False)
    print("Housing dataset saved to data/boston_housing.csv")

if __name__ == "__main__":
    ingest_data()