#!/usr/bin/python3
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import random

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x = x/255

index = random.choice(range(10000))

f, axes = plt.subplots(2, 10, sharey=True,figsize=(10,10))

for i, ax in enumerate(axes.flat):
    ax.imshow(x[i],cmap="gray")
# plt.show()

model = Sequential()
model.add(Conv2D(24, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(36, kernel_size=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

x = x.reshape((60000, 28, 28, 1))
y = y.reshape((60000, 1))
model.fit(x=x, y=y, epochs=3)

