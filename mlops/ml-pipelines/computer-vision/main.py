import mlflow.data.tensorflow_dataset
import tensorflow as tf
import mlflow
import mlflow.keras
import time
import mlflow.models.signature

from workflows.preprocessing import preprocessing_data
from workflows.training import train_model

from tensorflow.keras import models, layers
from mlflow.models import infer_signature

from prefect import flow

@flow(name='main-pipeline-xray-binary', version="1", description='Runnning Training Pipeline for Xray Binary Classifications')
def main():
    # Homework
    ## Need to check the refactor.
    ## Create validation pipeline
    ## Create data pipeline

    server_uri = 'http://localhost:8080' # Change to remote server if its needed
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment("xray-binary-classification")

    with mlflow.start_run() as run:
        mlflow.tensorflow.autolog()
        # Set ID
        # mlflow.log_params({
        #     'User': 'Ridha Ginanjar',
        #     "Time": time.strftime("%Y-%m-%d %H:%M:%S")
        # })


        zip_path = '/Users/dicoding/Dicoding/github/mlops/mlops/experiment-tracking/data/archive.zip'
        extracted_path = './data/'
        normal_class_dir = './data/chest_xray/train/NORMAL'
        augmented_dir = './data/chest_xray/train/NORMAL_augmented'

        # Preprocessing
        preprocessing_data(zip_path, extracted_path, normal_class_dir, augmented_dir)

        # # Set Dataset parameter
        # mlflow.log_param('dataset', 'chest-xray-v1')

        # Set notes about imbalanced dataset
        with open('note_preprocessing.txt', 'w') as f:
            f.write("The dataset contain imbalanced dataset problem where NORMAL class is less than PNEUMONIA class \n")
            f.write("The solutions to solve this problem is doing data augmentation for only NORMAL class")

        # Hyperparameter
        optimizer = 'adam'
        loss = 'binary_crossentropy'
        metrics=['accuracy']
        # mlflow.log_params({
        #     'optimizer': optimizer,
        #     'loss':loss,
        #     'metrics':metrics,
        # })

        # Training and testing directory
        train_dir = './data/chest_xray/train'
        test_dir = './data/chest_xray/test'

        # Training
        train_model(train_dir, test_dir, optimizer, loss, metrics)
        
        # Log input dataset
        # train_dataset = from_tensorflow(train_gen)
        # val_dataset = from_tensorflow(val_gen)
        # mlflow.log_input(train_dataset)
        # mlflow.log_input(val_dataset)
        
        # Set Evaluations
        # mlflow.log_metrics({
        #     'loss_val': loss_eval,
        #     'acc_val': acc_eval,
        #     'validation_accuracy': history.history['val_accuracy']
        # })

if __name__ == '__main__':
    main()