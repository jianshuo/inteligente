# 导入著名的 tensorflow。 在命令行下用 pip3 install tensorflow 安装
import tensorflow as tf
# 画图工具，人工智能学习必备
import matplotlib.pyplot as plt
# 从互联网上下载 mnist 数据集到本地 ~/.keras/datasets/mnist.npz
# x_train, y_train 分别是是60000个训练图像和答案
# x_test, y_test 分别是10000个测试图像和答案
# 训练的算是日常习题，测试的才是高考题。为了计算机防止作弊，计算机​读书的时候是不能看到高考试卷的
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 准备取出来64个图像画出来看看
c = 0
for index in range(850, 2000):
    # 不是 0 就算了
    if y_train[index] != 0:
        continue

    c += 1
    if c > 64:
        continue

    # 导入绘图工具 matplotlib。 在命令行下用 pip3 install matplotlib 安装
    # 把这个28x28的矩阵直接给 Matplot 画灰度​图
    ax = plt.subplot(8, 8, c)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(x_train[index], cmap="gray")
    print(index, c)
# 显示出来​
plt.show()
