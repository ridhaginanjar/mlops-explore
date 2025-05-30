import mlflow
import mlflow.exceptions
import datetime

from prefect import flow, task
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository


@task(name='check_metrics_models')
def checks_metrics_models(f1_score, pred_acc, metrics_regist_model):
    # Compare metrics between new model and registered model.
    # If better than registered, should be promote to new production tags (ready to use)

    print(f1_score, pred_acc)

    registered_f1 = metrics_regist_model.get('overall_f1')
    registered_pred_acc = metrics_regist_model.get('predict_acc')

    print(registered_f1, registered_pred_acc)

    if f1_score > registered_f1 and pred_acc > registered_pred_acc:
        return "promote"
    else:
        return "comparable"


@flow(name='update_registed_model')
def update_registered_model(run_id, f1_score, pred_acc):
    """
    This function has responsible to update existing model on MLFlow Registry. 
    If there are no existing model, the new model will be registered.
    """

    if f1_score is not None and pred_acc is not None:
        # Fetch the MLFlow client to interact with registry
        client = mlflow.tracking.MlflowClient()
        model_name = 'xray-binary-classif-staging'
        try:
            with mlflow.start_run(run_id=run_id) as run:
                model_versions = client.search_model_versions(f"name='{model_name}'")
                if model_versions:
                    # print(f"Model Version Info: {model_versions}")
                    # Model Version info object
                    # Model Version Info: [<ModelVersion: aliases=[], creation_timestamp=1747814744576, current_stage='None', description='', last_updated_timestamp=1747814744576, name='xray-binary-classification', run_id='af82cc86d9c3475a8c54a9b8ae0348cd', run_link='', source='mlflow-artifacts:/549623103595319352/af82cc86d9c3475a8c54a9b8ae0348cd/artifacts/model', status='READY', status_message=None, tags={}, user_id='', version='1'>]

                    registered_model = model_versions[0]
                    registered_name = registered_model.name
                    registered_version = registered_model.version
                    registered_run_id = registered_model.run_id

                    # print(f"Model registered tag {registered_model.tags}")
                    # registered_tag = registered_model.tags
                    # print(f"Model exists with version {registered_model.version}. Updating model ...")


                    # CHEK TEAMS, ini harus diimprove logicsnya.
                    # Set model tag to production 
                    latest_version = int(registered_version) + 1

                    # client.set_model_version_tag(model_name, str(latest_version), "validation_status", "pending")
                    # client.set_registered_model_alias(model_name,  "staging", latest_version)
                    # print(f"New model for {model_name} has set to STAGING and validation_status to PENDING")

                    # Continue to validation registered model.
                    run = client.get_run(registered_run_id)
                    metrics = run.data.metrics
                    
                    # Compare models
                    result = checks_metrics_models(f1_score, pred_acc, metrics)
                    print(result)

                    if result  =='promote':
                        print("New model performs better. Promoting to production...")

                        # New model
                        runs_uri = f"runs:/{run.info.run_id}/{model_name}"
                        model_src = RunsArtifactRepository.get_underlying_uri(runs_uri=runs_uri)
                        mv = client.create_model_version(name=model_name, source=model_src, run_id=run.info.run_id) # Update Versi Model

                        print(f"Name: {mv.name}")
                        print(f"Version: {mv.version}")
                        print(f"Description: {mv.description}")
                        print(f"Status: {mv.status}")


                        client.set_registered_model_alias(model_name, "production", latest_version) # Set new version to production alias
                        client.set_model_version_tag(model_name, str(latest_version), "validation_status", "promoted") # Set version tag promoted for new model version
                        client.set_model_version_tag(model_name, str(latest_version), "stage", "production") # Set new model tag to production
                        
                        # Old model
                        client.delete_registered_model_alias(model_name, "production", registered_version) # delete previous version from production alias
                        client.set_registered_model_alias(model_name, "archived", registered_version) # Set previous version to archived alias (for rollback)
                        client.delete_registered_model_tag(model_name, "validation_status", registered_version)
                        client.delete_registered_model_tag(model_name, "stage", registered_version)
                        client.set_registered_model_tag(model_name, "stage", "archived") # Set tag stage into archived


                        print(f"New {model_name} already promoted to productions")
                        print(f" Set validation_status to 'promoted' and alias 'production' for version {registered_version}")
                    else:
                        # If the new model is comparable, keep the current model in staging
                        mlflow.set_tag("new_model_version", latest_version)
                        mlflow.set_tag("validation_status", "comparable")
                        print("New model is comparable to the registered model. Stay in experiment and create new tag")
                else:
                    # Need to fix
                    # If no model exists, register a new model
                    print("No model found. Registering a new model...")
                    model_uri = f"runs:/{run.info.run_id}/model"
                    client.create_registered_model(model_name)  # Create a new registered model if none exists
                    client.create_model_version(model_name, model_uri, run.info.run_id)
                    print("New model version registered.")

        except mlflow.exceptions.MlflowException as e:
            print(f"Error while interacting with MLflow: {e}")
            raise
    else:
        print("Missing required parameters (loss, acc, or pred_acc). Model update aborted.")

# if __name__ == '__main__':
#     mlflow.set_tracking_uri("http://localhost:8080")
#     mlflow.set_experiment("xray-binary-classification")

#     run_id = "af82cc86d9c3475a8c54a9b8ae0348cd"
#     loss = 0.30335407555103302
#     acc = 0.9532938694953918
#     pred_acc = 0.7084615384615384
#     update_registered_model(run_id, loss, acc, pred_acc)



