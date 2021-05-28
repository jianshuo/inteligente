""" How AI recognize 0 - 9 """
import tensorflow

data = tensorflow.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = data

index = 100

print(x_train[100])
print(y_train[100])

