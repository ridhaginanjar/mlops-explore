import mlflow

from prefect import flow
from workflows.validations_flow.model_validation import classif_report, conf_matrix, auc_generator
from workflows.training import training_augmentations

@flow(name='model-validation')
def validate_model(run_id, y_true, y_pred, y_prob, train_dir, test_dir):

    with mlflow.start_run(run_id=run_id):
        mlflow.tensorflow.autolog()
        # Validasi Model
        ## Classification Report -> Precision, Recall, f1-score
        ## Confussion Matrix
        ## ROC
        ## AUC

        _, test_gen, _ = training_augmentations(train_dir, test_dir)

        # Classification Report
        classif_dict = classif_report(y_true,y_pred, target_names=test_gen.class_indices.keys())
        mlflow.log_metrics(classif_dict)

        # Confussion Matrix
        conf_matrix_dict = conf_matrix(y_true, y_pred, test_gen)
        mlflow.log_metrics(conf_matrix_dict)

        # ROC and AUC
        roc_auc_dict = auc_generator(y_true, y_prob)
        mlflow.log_metrics(roc_auc_dict)