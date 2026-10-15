import argparse
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

def deploy_model(alias):
    print(f"Deploying model with alias '{alias}' to environment...")
    model_uri = f"models:/DiseaseModel@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Successfully loaded and deployed model for alias '{alias}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", required=True, help="Model alias to deploy")
    args = parser.parse_args()
    deploy_model(args.alias)