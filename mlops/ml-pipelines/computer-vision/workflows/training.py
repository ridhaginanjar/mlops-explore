import mlflow

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

from mlflow.models import infer_signature

from prefect import flow, task

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


@task(name='prediction-test')
def predict_model(model, test_gen):
    # Prediction for validation
    y_true = test_gen.classes
    y_prob = model.predict(test_gen)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1)

    # Prediction for signature
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
    
    return y_true, y_prob, y_pred

    


@flow(name='training-pipeline', description='Run training process', log_prints=True)
def train_pipeline(train_dir, test_dir):

    with mlflow.start_run() as run:
        mlflow.tensorflow.autolog()
        run_id = run.info.run_id

        # Running Training Augmentations
        train_gen, test_gen, val_gen = training_augmentations(train_dir, test_dir)

        # Training
        callbacks = myCallBacks()

        # Hyperparameter
        OPTIMIZERS = 'adam'
        LOSSES = 'binary_crossentropy'
        METRICS=['accuracy']
        EPOCHS=20

        # Build Model
        model = build_model()

        # Set model summary
        with open("model_summary.txt", 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Running Process
        model.compile(optimizer=OPTIMIZERS, loss=LOSSES, metrics=METRICS)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs= EPOCHS,
            callbacks=[callbacks]
        )

        # Export Model
        # model_dir = './model'
        # export_model(model, model_dir)

        # Prediction
        y_true, y_prob, y_pred = predict_model(model, test_gen)
        
        # Evaluate
        # loss, acc = model.evaluate(test_gen)
        # print(f"Test accuracy: {acc: .4f}")

        # ------------------------------------------------
        # Archive Param
        
        # Set ID
        # mlflow.log_params({
        #     'User': 'Ridha Ginanjar',
        #     "Time": time.strftime("%Y-%m-%d %H:%M:%S")
        # })

        # # Set Dataset parameter
        # mlflow.log_param('dataset', 'chest-xray-v1')
                # mlflow.log_params({
        #     'optimizer': optimizer,
        #     'loss':loss,
        #     'metrics':metrics,
        # })

                
        # Set Evaluations
        # mlflow.log_metrics({
        #     'loss_val': loss_eval,
        #     'acc_val': acc_eval,
        #     'validation_accuracy': history.history['val_accuracy']
        # })

    return run_id, y_true, y_prob, y_pred