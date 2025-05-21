from prefect import flow

from utils.extract_zip import extract_zip
from utils.augmentations import imbalanced_augmentations

@flow(name='preprocessing', description='Run preprocessing data')
def preprocessing_data(zip_path, extracted_path, normal_class_dir, augmented_dir):

    # Procedure: Unzip Dataset
    extract_zip(zip_path, extracted_path)
    # Procedure: Data Augmentations
    imbalanced_augmentations(normal_class_dir, augmented_dir)

    pass