import mlflow
import io
import numpy as np

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

MLFLOW_TRACKING_URI = "http://localhost:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model(run_id):
    model_uri = f'runs:/{run_id}/model'
    loaded_model = mlflow.tensorflow.load_model(model_uri)

    return loaded_model


def preprocess_image(file, target_size=[150,150]):
    image = Image.open(io.BytesIO(file)).convert('RGB')
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Reshape to (1, 150, 150, 3)

    return image_array