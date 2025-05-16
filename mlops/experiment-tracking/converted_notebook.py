#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
import os
import zipfile
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras import models, layers


# # Importing Dataset

# Extract data into data path

extracted_data_path = 'data/'
zip_path = 'data/archive.zip'

with zipfile.ZipFile(zip_path, 'r') as zipfile:
    zipfile.extractall(extracted_data_path)

print(f"Data was extracted into {extracted_data_path} path") 


# Finding the directory informations
dataset_path = 'data/chest_xray'

for dirpath, _, filenames in os.walk(dataset_path):
    folder_name = os.path.basename(dirpath)
    file_count = len([f for f in filenames if not f.startswith(".")]) 
    print(f"Folder: {folder_name}, Banyaknya berkas: {file_count}")


# ## Problem - Imbalenced datasets

# ### Minority IDG

# Seperate the data based on labels -> only choose Normal (this is the minority)
minority_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)


normal_dir = './data/chest_xray/train/NORMAL'
aug_dir = './data/chest_xray/train/NORMAL_augmented'
os.makedirs(aug_dir,exist_ok=True)


images = os.listdir(normal_dir)
target_total = len(images) * 2 #Stopper bcs the objective is multiply the dataset.
generated = 0

for img_name in images:
    img_path = os.path.join(normal_dir, img_name)
    img = load_img(img_path, target_size=[150,150])
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape) # dibutuhkan karena kebutuhan .flow nanti formatnya (1,150,150,3)
    
    for batch in minority_aug.flow(img_array, batch_size=1, save_to_dir=aug_dir, save_prefix='aug', save_format='jpeg'):
        generated += 1
        if generated >= target_total:
            break
    if generated >= target_total:
        break


# ### Gabungkan dataset yang sudah augmented dengan direktori asli.

for fname in os.listdir(aug_dir):
    shutil.move(os.path.join(aug_dir, fname), os.path.join(normal_dir, fname))

print(f"Sebanyak {len(os.listdir(normal_dir))} data ditemukan pada berkas {normal_dir}")

shutil.rmtree(aug_dir)


# # Training
# ## Build ImageDataGenerator

# Tanpa augmentasi biar ga over.
train_datagen = ImageDataGenerator(
    rescale= 1/.255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1/.255,
)


# ## Build Generator for training & Validation

train_dir = './data/chest_xray/train'
test_dir = './data/chest_xray/test'

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

# ## Membangun Arsitektur

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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class myCallBacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.9):
            self.model.stop_training = True

CALLBACKS = myCallBacks()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[CALLBACKS]
)

loss, acc = model.evaluate(test_gen)
print(f"Test accuracy: {acc: .4f}")


# # Export Model to SavedModel
get_ipython().system('mkdir -p ./models/')
model.export("./models")

