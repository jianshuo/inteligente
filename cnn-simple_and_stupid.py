# 导入著名的 tensorflow。 在命令行下用 pip3 install tensorflow 安装
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
# 从互联网上下载 mnist 数据集到本地 ~/.keras/datasets/mnist.npz
# x_train, y_train 分别是是60000个训练图像和答案
# x_test, y_test 分别是10000个测试图像和答案
# 训练的算是日常习题，测试的才是高考题。为了计算机防止作弊，计算机​读书的时候是不能看到高考试卷的
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model = Sequential()
model.add(Flatten(input_shape = (28, 28, 1)))
model.add(Dense(10,  activation='softmax'))
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
model.fit(x_train, y_train)
