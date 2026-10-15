import sys
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")

def test_model(alias):
    model_uri = f"models:/DiseaseModel@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    
    data = pd.read_csv("data/disease_dataset.csv")
    X = data.drop("target", axis=1)
    y = data["target"]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Model test accuracy for alias '{alias}': {acc}")

    # Test condition threshold
    if acc < 0.70:
        print("Model accuracy is below threshold!")
        sys.exit(1)
    else:
        print("Model test passed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", required=True, help="Model alias to test")
    args = parser.parse_args()
    test_model(args.alias)