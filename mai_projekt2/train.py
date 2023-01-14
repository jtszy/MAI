import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

# POBIERANIE DANYCH TRENINGOWYCH
x_train = np.load("x_data.npy")
y_train = np.load("y_data.npy")

# STWORZENIE SIECI NEURONOWEJ

model = tf.keras.Sequential([
    tf.keras.layers.Dense(600, input_dim = 400, activation = "sigmoid"),
    tf.keras.layers.Dense(400, activation="tanh"),
    tf.keras.layers.Dense(300, activation = "softplus"),
    tf.keras.layers.Dense(200, activation = "sigmoid"),
    tf.keras.layers.Dense(100, activation = "tanh"),
    tf.keras.layers.Dense(4, activation="softplus")
])

model.compile(optimizer = "adamax", loss ="MSE")

model.fit(x_train, y_train, epochs = 100)

model.save_weights("model1_weights.h5")


