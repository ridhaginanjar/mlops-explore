import time
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

from prefect import flow, task


@task(name="check_metrics_models")
def checks_metrics_models(new_f1, new_acc, old_metrics):
    """Compare new model metrics with registered model metrics."""
    old_f1 = old_metrics.get("overall_f1", 0)
    old_acc = old_metrics.get("predict_acc", 0)

    print(f"[INFO] New F1: {new_f1}, New Accuracy: {new_acc}")
    print(f"[INFO] Registered F1: {old_f1}, Registered Accuracy: {old_acc}")

    if new_f1 > old_f1 and new_acc > old_acc:
        return "promote"
    return "comparable"


@flow(name="update_registered_model")
def update_registered_model(run_id, f1_score, pred_acc):
    """Register or update model in MLflow registry based on performance comparison."""

    if f1_score is None or pred_acc is None:
        print("[ERROR] Missing required metrics. Aborting update.")
        return

    model_name = "xray-binary-classification-nx"
    client = MlflowClient()

    try:
        with mlflow.start_run(run_id=run_id):
            model_uri = f"runs:/{run_id}/model"

            # Step 1: Check if model is already registered
            try:
                registered_model = client.get_registered_model(model_name)
                print(f"[INFO] Found registered model: {registered_model.name}")
                model_versions = client.search_model_versions(f"name='{model_name}'")

            except MlflowException:
                print("[INFO] No registered model found. Registering new model...")
                mv = mlflow.register_model(model_uri=model_uri, name=model_name)

                # Let registry catch up
                time.sleep(3)

                client.set_model_version_tag(model_name, mv.version, "validation_status", "promoted")
                client.set_model_version_tag(model_name, mv.version, "stage", "production")
                client.set_registered_model_alias(model_name, "production", mv.version)

                print(f"[SUCCESS] Model version {mv.version} registered and promoted to production.")
                return

            # Step 2: Compare new model vs latest version
            latest_model = sorted(model_versions, key=lambda m: int(m.version))[-1]
            reg_run = client.get_run(latest_model.run_id)
            reg_metrics = reg_run.data.metrics

            result = checks_metrics_models(f1_score, pred_acc, reg_metrics)

            if result == "promote":
                print("[INFO] New model is better. Promoting...")

                model_src = RunsArtifactRepository.get_underlying_uri(model_uri)
                mv = client.create_model_version(name=model_name, source=model_src, run_id=run_id)

                time.sleep(3)  # wait for MLflow registry consistency

                client.set_registered_model_alias(model_name, "production", mv.version)
                client.set_model_version_tag(model_name, mv.version, "validation_status", "promoted")
                client.set_model_version_tag(model_name, mv.version, "stage", "production")

                old_version = latest_model.version
                client.set_registered_model_alias(model_name, "archived", old_version)
                client.set_model_version_tag(model_name, old_version, "stage", "archived")

                print(f"[SUCCESS] New model promoted. Old version {old_version} archived.")

            else:
                print("[INFO] New model is not better. Keeping current production model.")
                mlflow.set_tag("validation_status", "comparable")
                mlflow.set_tag("comparison_result", "new model not promoted")

    except MlflowException as e:
        print(f"[MLFLOW ERROR] {e}")
        raise
