# 导入著名的 tensorflow。 在命令行下用 pip3 install tensorflow 安装
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization
# 从互联网上下载 mnist 数据集到本地 ~/.keras/datasets/mnist.npz
# x_train, y_train 分别是是60000个训练图像和答案
# x_test, y_test 分别是10000个测试图像和答案
# 训练的算是日常习题，测试的才是高考题。为了计算机防止作弊，计算机​读书的时候是不能看到高考试卷的
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
x_train = x_train.astype(np.float32).reshape(*x_train.shape, 1)
# x_train /= 255
# # y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype=np.float32)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
model.summary()
model.fit(x_train, y_train, epochs=20)
