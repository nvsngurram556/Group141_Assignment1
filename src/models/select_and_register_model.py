import mlflow
from mlflow.tracking import MlflowClient


def select_best_and_register(experiment_name: str, registry_name: str = "HousingModel"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.rmse ASC"],
        max_results=10
    )
    if not runs:
        raise RuntimeError("No runs found")

    best = runs[0]
    print("Best run id:", best.info.run_id)
    # assume model artifact saved under 'model'
    model_source = f"runs:/{best.info.run_id}/model"

    # register model
    result = mlflow.register_model(model_source, registry_name)
    print("Registered model version:", result.version)

    # transition to Staging (optional)
    client.transition_model_version_stage(name=registry_name, version=result.version, stage="Staging")
    print(f"Model {registry_name} version {result.version} transitioned to Staging")


if __name__ == "__main__":
    select_best_and_register("california-housing", "HousingModel")
