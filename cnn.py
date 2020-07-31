import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
import matplotlib.pyplot as plt
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape) # (60000, 28, 28)

x_train = x_train.reshape(*x_train.shape, 1).astype(np.float32)
x_test = x_test.reshape(*x_test.shape, 1).astype(np.float32)

x_train /= 255
x_test /= 255

print(x_train.shape)
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(196, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=1)
# model.save_weights('mnist.h5')
# model.load_weights('mnist.h5')
print(model.summary())
print('Evaluate', model.evaluate(x_test, y_test))
pos = random.choice(range(60000))
rows = 8
cols = 4

feature_map = model.predict(x_train[pos:pos+1])
print('Truth', y_train[pos:pos+1])
print('Feature map shape', feature_map.shape)
print('Predict is', np.argmax(feature_map, axis=-1))


md = Model(inputs=model.inputs, outputs=model.layers[10].output)
for i in range(16):
    plt.subplot(rows, cols, i+1)
    for layer in md.layers:
        if layer.name == 'conv2d':
            filters, biases = layer.get_weights()
            print('Final dense filter shape', filters.shape)
            plt.imshow(filters[:, :, 0, i], cmap='gray')
prds = md.predict(x_train[pos:pos+1])
print('prediction shape', prds.shape)

for i in range(16):
    plt.subplot(rows, cols, i+17)
    print('predction shape', prds[0, :, :, i].shape)
    plt.imshow(prds[0, :, :, i], cmap='gray')

plt.show()

exit()
plt.subplot(rows, cols, 1)
plt.imshow(x_train[pos][:, :, 0], cmap='gray')

plt.subplot(rows, cols, 2)
md = Model(inputs=model.inputs, outputs=model.layers[0].output)
print(md.summary())
prd = md.predict(x_train[pos:pos+1])
print('prediction shape', prd.shape)
img = prd[0, :, :, 0]
plt.imshow(img, cmap='gray')

plt.subplot(rows, cols, 3)
md = Model(inputs=model.inputs, outputs=model.layers[1].output)
print(md.summary())
prd = md.predict(x_train[pos:pos+1])
print('prediction shape', prd.shape)
img = prd[0, :, :, 0]
plt.imshow(img, cmap='gray')

plt.subplot(rows, cols, 4)
md = Model(inputs=model.inputs, outputs=model.layers[2].output)
print(md.summary())
prd = md.predict(x_train[pos:pos+1])
print('prediction shape', prd.shape)
img = prd[0]
plt.bar(range(196), img)

plt.subplot(rows, cols, 5)
md = Model(inputs=model.inputs, outputs=model.layers[3].output)
print(md.summary())
prd = md.predict(x_train[pos:pos+1])
print('prediction shape', prd.shape)
img = prd[0]
plt.bar(range(10), img)
#

plt.subplot(rows, cols, 6)
print('Len of layers', len(md.layers))
filters, biases = md.layers[1].get_weights()
print(filters.shape)
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min)/(f_max - f_min)
print('filter shape', filters.shape)
plt.imshow(filters[:, :, 0, 0], cmap='gray')

plt.subplot(rows, cols, 7)
for layer in md.layers:
    if layer.name == 'dense':
        filters, biases = layer.get_weights()
        print('Final dense filter shape', filters.shape)
        plt.imshow(filters, cmap='gray')
plt.show()