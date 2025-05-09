#!/usr/bin/env python
# coding: utf-8

# # Time Series Analysis
# 
# Kasus ini akan melakukan time series analysis dari dataset bernama Daily Climate Series.
# Link Dataset: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
# 
# Kasusnya adalah ingin memprediksi cuaca.
# 
# Notes: 
# Belum ada info diawalain itu memprediksi di satu langkah kedepan (1 hari) atau beberapa langkah kedepan

# # Importing Library
import mlflow.experiments
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf

import mlflow
mlflow.tensorflow.autolog()

# # Importing Dataset
df = pd.read_csv("https://github.com/ridhaginanjar/mlops-explore/releases/download/climate-time-series/DailyDelhiClimateTrain.csv")
df.head()


# # Cleaning Dataset

# ## Cleansing
# Checking null values
df.isnull().sum()


# ## Feature Selection
dates = df['date'].values
temp = df['meantemp'].values

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1) # Menambah dimensi untuk series -> e.g: [1,2,3,4] menjadi [[1],[2],[3],[4],[5]]
    ds = tf.data.Dataset.from_tensor_slices(series) # Mengubah series menjadi dataset tensorflow
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True) # Melakukan windowing (dalam kasus ini 60 +1)
    ds = ds.flat_map(lambda x:x.batch(window_size +1)) # Melakukan flattening -> e.g: [[1],[2],[3],[4],[5]] menjadi [1,2,3] [2,3,4] [3,4,5]
    ds = ds.shuffle(shuffle_buffer) # Mengacak urutan window -> [2,3,4] [1,2,3] [3,4,5]
    ds = ds.map(lambda x: (x[:-1], x[-1:])) # memecah x dan y -> x = [[2,3]] y =[[4]]
    return ds.batch(batch_size).prefetch(1) # mengelompokkan per batch -> e.g: batch 1 x = [[2,3]] y =[[4]], batch 2 = xxxx 


# # Training
train_set = windowed_dataset(temp, window_size=60, batch_size=100, shuffle_buffer=1000)

mlflow.set_tracking_uri("http://35.193.47.166:5000")

with mlflow.start_run() as run:
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

optimizer = tf.keras.optimizers.SGD(learning_rate=1.000e-04, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
history = model.fit(train_set, epochs=100)


# # Evaluate

forecast = history.model.predict(train_set)