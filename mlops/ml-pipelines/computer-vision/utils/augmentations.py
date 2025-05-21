import tensorflow as tf
import os
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from prefect import task

@task(name='merge-imbalanced-augment', description='Merge The Results of Augmentations on Imbalenced Dataset')
def merged_imbalanced_augment(aug_dir, normal_dir):
     # Merge Augmented Dataset into original directory (NORMAL training directory)
    for fname in os.listdir(aug_dir):
        shutil.move(os.path.join(aug_dir, fname), os.path.join(normal_dir, fname))
        print(f"Sebanyak {len(os.listdir(normal_dir))} data ditemukan pada berkas {normal_dir}")

    # Delete Augmented Directory
    shutil.rmtree(aug_dir)


@task(name='imbalanced-augmentations', description='Data Augmentation for NORMAL class (Minority and imbalanced)')
def imbalanced_augmentations(normal_dir, aug_dir):
    RESCALE=1./255
    ROTATION_RANGE=10
    WIDTH_SHIFT_RANGE=0.1
    HEIGHT_SHIFT_RANGE=0.1
    ZOOM_RANGE=0.1
    BRIGHTNESS_RANGE=[0.8, 1.2]

    minority_aug = ImageDataGenerator(
        rescale=RESCALE,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        brightness_range=BRIGHTNESS_RANGE
    )

    os.makedirs(aug_dir, exist_ok=True)

    # Data Augmentation for only NORMAL Dataset
    images = os.listdir(normal_dir)
    target_total = len(images) * 2 #Stopper bcs the objective is multiply the dataset.
    generated = 0

    for img_name in images:
        img_path = os.path.join(normal_dir, img_name)
        img = load_img(img_path, target_size=[150,150])
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape) # dibutuhkan karena kebutuhan .flow nanti formatnya (1,150,150,3)
        
        for _ in minority_aug.flow(img_array, batch_size=1, save_to_dir=aug_dir, save_prefix='aug', save_format='jpeg'):
            generated += 1
            if generated >= target_total:
                break
        if generated >= target_total:
            break

    # Procedure: Merge Augmented Dataset into original directory (NORMAL training directory)
    merged_imbalanced_augment(aug_dir, normal_dir)
    print("NORMAL class has been augmented successfully!")


@task(name='training-idg-for-training', description='Running Image Data Generator for training')
def training_augmentations(train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale= 1/.255,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(
        rescale=1/.255,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        subset='training',
        class_mode='binary',
        target_size=[150,150],
        batch_size=32,
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        subset='validation',
        target_size=[150,150],
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=[150,150],
        batch_size=32,
        class_mode='binary',
        shuffle=False # because important to evaluate.
    )

    return train_gen, test_gen, val_gen