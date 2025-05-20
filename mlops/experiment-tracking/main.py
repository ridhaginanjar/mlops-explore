import mlflow.data.tensorflow_dataset
import tensorflow as tf
import mlflow
import mlflow.keras
import time
import mlflow.models.signature

from utils.callbacks import myCallBacks
from utils.export_model import export_model
from utils.extract_zip import extract_zip

from workflows.preprocessing import data_augmentations
from workflows.training import train
from workflows.evaluations import evaluate_model

from tensorflow.keras import models, layers
from mlflow.models import infer_signature
from mlflow.data.tensorflow_dataset import from_tensorflow


def main():
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

        # Extract zipfile into our data folder.
        extract_zip(zip_path, extracted_path)

        # Augmented minority dataset (only for NORMAL class)
        data_augmentations(normal_class_dir, augmented_dir)

        # # Set Dataset parameter
        # mlflow.log_param('dataset', 'chest-xray-v1')

        # Set notes about imbalanced dataset
        with open('note_preprocessing.txt', 'w') as f:
            f.write("The dataset contain imbalanced dataset problem where NORMAL class is less than PNEUMONIA class \n")
            f.write("The solutions to solve this problem is doing data augmentation for only NORMAL class")

        # Model Arsitektur and configurations
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=[150,150,3]),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        # Set model summary
        with open("model_summary.txt", 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Hyperparameter
        optimizer = 'adam'
        loss = 'binary_crossentropy'
        metrics=['accuracy']
        CALLBACKS = myCallBacks()
        # mlflow.log_params({
        #     'optimizer': optimizer,
        #     'loss':loss,
        #     'metrics':metrics,
        # })

        # Training and testing directory
        train_dir = './data/chest_xray/train'
        test_dir = './data/chest_xray/test'

        # Training
        model_trained, train_gen, val_gen, test_gen, history = train(train_dir, test_dir, 
                                                model, optimizer, loss, metrics,
                                                callbacks=CALLBACKS)
        
        # Log input dataset
        # train_dataset = from_tensorflow(train_gen)
        # val_dataset = from_tensorflow(val_gen)
        # mlflow.log_input(train_dataset)
        # mlflow.log_input(val_dataset)

        # Evaluate
        loss_eval, acc_eval = evaluate_model(model_trained, test_gen)
        
        # Set Evaluations
        # mlflow.log_metrics({
        #     'loss_val': loss_eval,
        #     'acc_val': acc_eval,
        #     'validation_accuracy': history.history['val_accuracy']
        # })

        # Export Model
        model_dir = './model'
        export_model(model_trained, model_dir)

        ## Log model with the signature (signature is used to find the format for input-output model)
        x_test_batch, _ = next(test_gen)
        x_sample = x_test_batch[:1] # Take one image

        ## Predict with one sample image data test
        pred = model.predict(x_sample)

        ## Log signature
        signature = infer_signature(x_sample, pred)
        # mlflow.tensorflow.log_model(model
        #                             ,artifact_path='model', signature=signature, pip_requirements='/Users/dicoding/Dicoding/github/mlops/requirements.txt')

if __name__ == '__main__':
    main()