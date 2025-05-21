import mlflow

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

from mlflow.models import infer_signature

from prefect import flow

from utils.augmentations import training_augmentations
from utils.callbacks import myCallBacks
from utils.export_model import export_model

def build_model():
    return models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=[150,150,3]),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])


@flow(name='training-flow', description='Run training process', log_prints=True)
def train_model(train_dir, test_dir, OPTIMIZERS: str, LOSSES: str, METRICS: list, EPOCHS=20):

    # Running Training Augmentations
    train_gen, test_gen, val_gen = training_augmentations(train_dir, test_dir)

    # Training
    callbacks = myCallBacks()
    
    # Build Model
    model = build_model()
    # Set model summary
    with open("model_summary.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.compile(optimizer=OPTIMIZERS, loss=LOSSES, metrics=METRICS)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs= EPOCHS,
        callbacks=[callbacks]
    )

    # Export Model
    model_dir = './model'
    export_model(model, model_dir)

    # Prediction
    ## Log model with the signature (signature is used to find the format for input-output model)
    x_test_batch, _ = next(test_gen)
    x_sample = x_test_batch[:1] # Take one image

    ## Predict with one sample image data test
    pred = model.predict(x_sample)

    ## Log signature
    signature = infer_signature(x_sample, pred)
    mlflow.tensorflow.log_model(model
                                ,artifact_path='model', signature=signature, 
                                pip_requirements='/Users/dicoding/Dicoding/github/mlops/requirements.txt')
    
    # Evaluate
    loss, acc = model.evaluate(test_gen)
    print(f"Test accuracy: {acc: .4f}")