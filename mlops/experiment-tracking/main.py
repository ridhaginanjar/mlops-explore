import tensorflow as tf

from utils.callbacks import myCallBacks
from utils.export_model import export_model
from utils.extract_zip import extract_zip

from workflows.preprocessing import data_augmentations
from workflows.training import train
from workflows.evaluations import evaluate_model

from tensorflow.keras import models, layers


def main():
    zip_path = './data/archive.zip'
    extracted_path = './data/'
    normal_class_dir = './data/chest_xray/train/NORMAL'
    augmented_dir = './data/chest_xray/train/NORMAL_augmented'

    # Extract zipfile into our data folder.
    extract_zip(zip_path, extracted_path)

    # Augmented minority dataset (only for NORMAL class)
    data_augmentations(normal_class_dir, augmented_dir)

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

    # Hyperparameter
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics=['accuracy']
    CALLBACKS = myCallBacks()

    # Training and testing directory
    train_dir = './data/chest_xray/train'
    test_dir = './data/chest_xray/test'

    # Training
    model_trained, _, test_gen = train(train_dir, test_dir, 
                                               model, optimizer, loss, metrics,
                                               callbacks=CALLBACKS)

    # Evaluate
    evaluate_model(model_trained, test_gen)

    # Export Model
    model_dir = './model'
    export_model(model_trained, model_dir)

if __name__ == '__main__':
    main()