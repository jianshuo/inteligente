# 导入著名的 tensorflow。 在命令行下用 pip3 install tensorflow 安装
import tensorflow as tf
import numpy as np
# 从互联网上下载 mnist 数据集到本地 ~/.keras/datasets/mnist.npz
# x_train, y_train 分别是是60000个训练图像和答案
# x_test, y_test 分别是10000个测试图像和答案
# 训练的算是日常习题，测试的才是高考题。为了计算机防止作弊，计算机​读书的时候是不能看到高考试卷的
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
index = 1000
# 把第1001个数字的图像打印出来
print(x_train[index])
# 把“正确答案”打印出来。结果应该是 0
print(y_train[index])

# 导入绘图工具 matplotlib。 在命令行下用 pip3 install matplotlib 安装
import matplotlib.pyplot as plt
# 把这个28x28的矩阵直接给 Matplot 画灰度​图
plt.imshow(x_train[index], cmap="gray")
# 显示出来​
plt.show()



def filter(a, kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])):
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

f = np.array([[-1.0, 0.0, 1.1],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]])
filtered = filter(x_train[index])
plt.imshow(filtered, cmap="gray")
# 显示出来​
plt.show()
print(filtered.shape)

from tensorflow.keras import layers

layer = layers.MaxPooling2D(pool_size=(10,10))

layerAct = layers.Activation(activation='relu')
outputs = layerAct(filtered.reshape(1, 28, 28, 1))
plt.imshow(outputs[0, :, :, 0], cmap="gray")
plt.show()

inputs = tf.random.uniform(shape=(10, 20, 1, 1))
outputs = layer(filtered.reshape(1, 28, 28, 1))
plt.imshow(outputs[0, :, :, 0], cmap="gray")

