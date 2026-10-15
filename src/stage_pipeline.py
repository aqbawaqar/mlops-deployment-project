import argparse
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def handle_stage(action):
    if action == "pre_prod_start":
        # Load 'Challenger' and assign alias 'Challenger-pre-test'
        model_version = client.get_model_version_by_alias("DiseaseModel", "Challenger")
        client.set_registered_model_alias("DiseaseModel", "Challenger-pre-test", model_version.version)
        print(f"Version {model_version.version} assigned alias 'Challenger-pre-test'")

    elif action == "pre_prod_success":
        # Update alias to 'Challenger-post-test'
        model_version = client.get_model_version_by_alias("DiseaseModel", "Challenger-pre-test")
        client.set_registered_model_alias("DiseaseModel", "Challenger-post-test", model_version.version)
        print(f"Version {model_version.version} updated to alias 'Challenger-post-test'")

    elif action == "prod":
        # Load 'Challenger-post-test' and assign alias 'Champion'
        model_version = client.get_model_version_by_alias("DiseaseModel", "Challenger-post-test")
        client.set_registered_model_alias("DiseaseModel", "Champion", model_version.version)
        print(f"Version {model_version.version} promoted to alias 'Champion'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["pre_prod_start", "pre_prod_success", "prod"])
    args = parser.parse_args()
    handle_stage(args.action)