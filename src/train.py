import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_Deployment_Experiment")

def train_and_register_dev():
    data = pd.read_csv("data/disease_dataset.csv")
    X = data.drop("target", axis=1)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_type", "RandomForest")

        # Register Model in MLflow Registry
        model_info = mlflow.sklearn.log_model(model, "model", registered_model_name="DiseaseModel")

        # Assign Alias 'Challenger'
        client = MlflowClient()
        client.set_registered_model_alias("DiseaseModel", "Challenger", model_info.registered_model_version)
        print(f"Model version {model_info.registered_model_version} registered with alias 'Challenger'.")

if __name__ == "__main__":
    train_and_register_dev()