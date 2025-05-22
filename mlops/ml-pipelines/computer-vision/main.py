import mlflow.data.tensorflow_dataset
import tensorflow as tf
import mlflow
import mlflow.keras
import time
import mlflow.models.signature

from workflows.preprocessing import preprocessing_data
from workflows.training import train_pipeline
from workflows.validation import validate_model

from tensorflow.keras import models, layers
from mlflow.models import infer_signature

from prefect import flow

@flow(name='main-pipeline-xray-binary', version="1", description='Runnning Pipeline for Xray Binary Classifications')
def main_pipeline():
    # Homework
    ## Need to check the refactor.
    ## Create validation pipeline
    ## Create data pipeline

    server_uri = 'http://localhost:8080' # Change to remote server if its needed
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment("xray-binary-classification")

    zip_path = '/Users/dicoding/Dicoding/github/mlops/mlops/experiment-tracking/data/archive.zip'
    extracted_path = './data/'
    normal_class_dir = './data/chest_xray/train/NORMAL'
    augmented_dir = './data/chest_xray/train/NORMAL_augmented'

    # Preprocessing Data -> Could be data pipeline
    preprocessing_data(zip_path, extracted_path, normal_class_dir, augmented_dir)

    # Set notes about imbalanced dataset
    with open('note_preprocessing.txt', 'w') as f:
        f.write("The dataset contain imbalanced dataset problem where NORMAL class is less than PNEUMONIA class \n")
        f.write("The solutions to solve this problem is doing data augmentation for only NORMAL class")

    # Training and testing directory
    train_dir = './data/chest_xray/train'
    test_dir = './data/chest_xray/test'

    # Training Pipeline
    run_id, y_true, y_prob, y_pred = train_pipeline(train_dir, test_dir)

    # Model Validation -> part of model validation
    validate_model(run_id, y_true, y_pred, y_prob, train_dir, test_dir)
    
    
    # Log input dataset
    # train_dataset = from_tensorflow(train_gen)
    # val_dataset = from_tensorflow(val_gen)
    # mlflow.log_input(train_dataset)
    # mlflow.log_input(val_dataset)

if __name__ == '__main__':
    main_pipeline()