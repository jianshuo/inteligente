# 导入著名的 tensorflow。 在命令行下用 pip3 install tensorflow 安装
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
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
model.fit(x_train, y_train, epochs=20)
# model.save_weights('mnist.h5')
model.load_weights('mnist.h5')

import random
import matplotlib.pyplot as plt
pos = random.choice(range(60000))
pos = 1002
plt.imshow(x_train[pos][:,:,0], cmap='gray')
plt.show()
#
# md = Model(inputs=model.inputs, outputs=model.layers[0].output)
# for layer in md.layers:
#     print('layer name:', layer.name)
#     if layer.name == 'conv2d':
#         filters, biases = layer.get_weights()
#         print('filter is', filters)
#         print('bias is', biases)
#         print('Conv2D filter shape', filters.shape)
#
#         rows = 8
#         cols = 8
#         for i in range(32):
#             ax = plt.subplot(rows, cols, i+1)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             plt.imshow(filters[:, :, 0, i], cmap='gray')
#
# prds = md.predict(x_train[pos:pos+1])
# print('prediction shape', prds.shape)
#
# for i in range(32):
#     ax = plt.subplot(rows, cols, i+33)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     print('predction shape', prds[0, :, :, i].shape)
#     plt.imshow(prds[0, :, :, i], cmap='gray')
# plt.show()
#
# exit()



rows = 8
cols = 4
# 把输入的原图画出来

# 第一层 Conv2D 网络
md = Model(inputs=model.inputs, outputs=model.layers[0].output)
filters, bias = model.layers[0].get_weights()
print(filters.shape)
for index in range(32):
    ax = plt.subplot(8, 4, index+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(filters[:, :, 0, index], cmap='gray')
plt.show()
print(filters, bias)
print(md.summary())
prd = md.predict(x_train[pos:pos+1])
print('prediction shape', prd.shape)

for index in range(32):
    ax = plt.subplot(rows, cols, index+1)
    ax.set_xticks([])
    ax.set_yticks([])
    img = prd[0, :, :, index]
    print(img)
    plt.imshow(img, cmap='gray')
plt.show()

for index in range(32):
    ax = plt.subplot(rows, cols, index+1)
    ax.set_xticks([])
    ax.set_yticks([])
    md = Model(inputs=model.inputs, outputs=model.layers[1].output)
    print(md.summary())
    prd = md.predict(x_train[pos:pos+1])
    print('prediction shape', prd.shape)
    img = prd[0, :, :, index]
    plt.imshow(img, cmap='gray')
plt.show()

md = Model(inputs=model.inputs, outputs=model.layers[2].output)
print(md.summary())
prd = md.predict(x_train[pos:pos+1])
print('prediction shape', prd.shape)
img = prd[0, :]
plt.bar(range(5408), img)
plt.show()

md = Model(inputs=model.inputs, outputs=model.layers[3].output)
print(md.summary())
prd = md.predict(x_train[pos:pos+1])
print('prediction shape', prd.shape)
img = prd[0, :]
plt.bar(range(10), img)
plt.show()

print(img)

[0.0000000e+00 1.0000000e+00 0.0000000e+00 3.3420391e-30 1.6909421e-26
 7.0893234e-33 0.0000000e+00 0.0000000e+00 5.1055432e-27 3.4242518e-33]


# exit()
#
# # plt.bar(range(196), img)
#
# plt.subplot(rows, cols, 5)
# md = Model(inputs=model.inputs, outputs=model.layers[5].output)
# print(md.summary())
# prd = md.predict(x_train[pos:pos+1])
# print('prediction shape', prd.shape)
# img = prd[0]
# plt.bar(range(28), img)
#
# plt.subplot(rows, cols, 6)
# md = Model(inputs=model.inputs, outputs=model.layers[6].output)
# print(md.summary())
# prd = md.predict(x_train[pos:pos+1])
# print('prediction shape', prd.shape)
# img = prd[0]
# plt.bar(range(10), img)
#
# plt.show()
