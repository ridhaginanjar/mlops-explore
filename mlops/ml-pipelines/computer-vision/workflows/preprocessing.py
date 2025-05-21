import tensorflow as tf
import os
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def data_augmentations(normal_dir, aug_dir):
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

    # Merge Augmented Dataset into original directory (NORMAL training directory)
    for fname in os.listdir(aug_dir):
        shutil.move(os.path.join(aug_dir, fname), os.path.join(normal_dir, fname))
        print(f"Sebanyak {len(os.listdir(normal_dir))} data ditemukan pada berkas {normal_dir}")

    shutil.rmtree(aug_dir)
