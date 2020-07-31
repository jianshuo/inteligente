import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import random
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)




# 咱给变成4维的
x_train = x_train.reshape(*x_train.shape, 1).astype('float32')
x_test = x_test.reshape(*x_test.shape, 1).astype('float32')

input_shape = (28, 28, 1)

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(30, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Visualize all the layers
print(model.summary())

for layer in model.layers:
    print(layer.name)
    print(layer)

    if layer.name == 'conv2d_1':
        filters, biases = layer.get_weights()
    
        fmin, fmax = filters.min(), filters.max()
        filters = (filters - fmin) / (fmax - fmin)
        print(filters.round(1))
        print('filter shape', filters.shape)

        ix = 1
        for i in range(6):
            plt.imshow(filters[:,:,0,0], cmap='gray')
            print(filters[:,:,0,0])
            plt.show()

        # plot first few filters
        n_filters, ix = 6, 1
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            print(f'Filter # {i}', f)
            # plot each channel separately
            for j in range(0):
                # specify subplot and turn of axis
                ax = plt.subplot(n_filters, 3, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                print('j', j, f.shape)
                plt.imshow(f[:, :, j], cmap='gray')
                ix += 1
        # show the figure
        plt.show()

model.fit(x=x_train,y=y_train, epochs=10)
result = model.evaluate(x_test, y_test)
print(result)
image_index = random.choice(range(0, 10000))
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print('predict', pred.argmax(), 'fact', y_test[image_index])
