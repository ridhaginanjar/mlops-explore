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
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf


# # Importing Dataset
df = pd.read_csv("https://github.com/ridhaginanjar/mlops-explore/releases/download/household-time-series/household_power_consumption.csv", sep=',', infer_datetime_format=True, index_col='datetime', header=0)
df.head()


# # Cleaning Dataset
# Normalization
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

data = df.values
data = normalize_series(data, data.min(axis=0), data.max(axis=0))


# ## Feature Selection
N_FEATURES = len(df.columns)
SPLIT_TIME = int(len(data) * 0.5)
x_train = data[:SPLIT_TIME]
x_valid = data[SPLIT_TIME:]

def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series) # Mengubah series menjadi dataset tensorflow
    ds = ds.window(n_past + n_future, shift=shift, drop_remainder=True) # Melakukan windowing (dalam kasus ini 60 +1)
    ds = ds.flat_map(lambda x:x.batch(n_past + n_future)) # Melakukan flattening -> e.g: [[1],[2],[3],[4],[5]] menjadi [1,2,3] [2,3,4] [3,4,5]
    ds = ds.map(lambda x: (x[:n_past], x[n_future:])) # memecah x dan y -> x = [[2,3]] y =[[4]]
    return ds.batch(batch_size).prefetch(1) # mengelompokkan per batch -> e.g: batch 1 x = [[2,3]] y =[[4]], batch 2 = xxxx 


# ## Splitting Data

BATCH_SIZE = 32
N_PAST = 24
N_FUTURE = 24
SHIFT = 1

train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)
valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)


# # Training
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(N_PAST, N_FEATURES)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(N_FEATURES)
])

class myCallBacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('mae') < 0.555 and logs.get('val_mae') < 0.555):
            self.model.stop_training = True

callbacks = myCallBacks()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mae', optimizer=optimizer, metrics=["mae"])


history = model.fit(train_set, validation_data=(valid_set), epochs=100, callbacks=callbacks, verbose=1)

train_pred = model.predict(train_set)
train_pred[0][0]