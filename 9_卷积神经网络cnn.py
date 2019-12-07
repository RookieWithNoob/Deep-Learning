import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 参数: 名字 默认值 说明
tf.app.flags.DEFINE_integer("max_step", 2000, "模型训练的步数")
tf.app.flags.DEFINE_string("data_dir", "./mnist/input_data", "数据文件路径")
tf.app.flags.DEFINE_string("model_dir", "./ann/model", "模型文件的保存和加载路径")
tf.app.flags.DEFINE_integer("is_train", 1, "模型训练")

# 定义获取命令行参数名字
FLAGS = tf.app.flags.FLAGS


class Ann():
    """手写数字案例"""

    def __init__(self, mnist):
        self.mnist = mnist
        self.weight = 784
        self.bias = 10
        self.labels = 10

    def full_connected(self):
        # 1. 建立数据的占位符 x [None, 784] y[None, 10]
        with tf.variable_scope("data"):
            x = tf.placeholder(tf.float32, shape=(None, self.weight))

            y_true = tf.placeholder(tf.int32, shape=(None, self.labels))

        # 2. 建立一个全连接层的神经网络 权重[784, 10] 偏置 [10]
        with tf.variable_scope("fc_model"):
            # 随机初始化权重和偏置
            weight = tf.Variable(tf.random_normal(shape=(self.weight, self.labels), mean=0.0, stddev=1.0),
                                 name="quanzhong")
            bias = tf.Variable(tf.zeros(shape=(self.bias), name="pianzhi"))

            # 3.预测none个样本的输出结果
            # [None, 784]*[784, 10] + [10]
            y_predict = tf.matmul(x, weight) + bias

        # 4. 计算所有样本的交叉熵损失 然后求平均值
        with tf.variable_scope("soft_cross"):
            # none个损失值的列表
            loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
            # 求列表损失值平均值
            loss = tf.reduce_mean(loss_list)

        # 5. 梯度下降优化求出损失
        with tf.variable_scope("optimizer"):
            train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

        # 6. 计算准确率
        with tf.variable_scope("acc"):
            equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
            # equal_list None个样本
            # [1, 0, 1, 1, 0, 1] 求平均值
            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

        # 收集tensor (变量) 损失值 权重 准确率 偏置
        # 单值
        tf.summary.scalar(name="losses", tensor=loss)
        tf.summary.scalar(name="zhunquelv", tensor=accuracy)
        # 多维 1 2 3
        tf.summary.histogram(name="pianzhi", values=bias)
        tf.summary.histogram(name="weights", values=weight)

        # . 定义合并tensor 的op
        merged = tf.summary.merge_all()
        # 定义一个初始化变量的op
        init_op = tf.global_variables_initializer()
        # . 定义一个保存模型的实例
        saver = tf.train.Saver()

        # 7. 开启会话去训练
        with tf.Session() as sess:
            # 初始化变量
            sess.run(init_op)

            # 建立事件文件
            filewriter = tf.summary.FileWriter("./summary/ann", graph=sess.graph)

            if FLAGS.is_train == 1:
                # 迭代步数去训练 更新参数预测
                for i in range(FLAGS.max_step):
                    # 取出真实存在的特征值和目标值
                    mnist_x, mnist_y = self.mnist.test.next_batch(100)
                    # 运行训练op
                    sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                    # 运行合并的tensor
                    summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                    filewriter.add_summary(summary, i)

                    print("训练第%d步,准确率为%f,损失值为%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}),
                                                    sess.run(loss, feed_dict={x: mnist_x, y_true: mnist_y})))

                # 保存模型
                saver.save(sess, FLAGS.model_dir)

            else:
                for i in range(100):
                    # 在训练之前加载模型 覆盖参数 从上次训练的参数结果开始   判断文件是否存在
                    if os.path.exists("./ann/checkpoint"):
                        # 存在则加载模型
                        saver.restore(sess, FLAGS.model_dir)

                    # 每次测试一张图片 [0,0,0,0,1,0,0,0,0,0]
                    x_test, y_test = self.mnist.test.next_batch(1)  # 特征值 目标值
                    # 预测
                    print("测试的第%d张图片,数字是 %d,预测结果是 %d" % (
                        i,
                        tf.argmax(y_test, 1).eval(),
                        tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()
                    ))

        return None

    def weight_variables(self, shape):
        """
        定义一个初始化权重的函数
        :param shape: 权重形状
        :return:
        """
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))

        return w

    def bias_variables(self, shape):
        """
        定义一个初始化偏置的函数
        :param shape: 偏置形状
        :return:
        """
        b = tf.Variable(tf.constant(0.0, shape=shape))

        return b

    def model(self):
        """
        自定义的卷积模型
        :return:
        """
        # 1. 准备数据的占位符 x[None, 784] y_true[None,10]
        with tf.variable_scope("data"):
            x = tf.placeholder(tf.float32, shape=(None, self.weight))

            y_true = tf.placeholder(tf.int32, shape=(None, self.labels))

        # 2. 一卷积层 卷积 激活 池化
        with tf.variable_scope("conv_1"):
            # 随机初始化权重 5*5的窗口 黑白1张表 32个人 步长1
            w_conv1 = self.weight_variables((5, 5, 1, 32))
            # 随机初始化偏置 偏置和多少个人是一样的
            b_conv1 = self.bias_variables((32,))

            # 1. 卷积 5*5的窗口 黑白1张表 32个人 步长1
            # 先对x进行形状改变[None, 984]--->[None, 28, 28, 1] 不知道的填-1
            x_reshape = tf.reshape(x, shape=(-1, 28, 28, 1))
            # [None, 28, 28, 1]---->[None,28,28,32]
            # 卷积+激活
            x_relu1 = tf.nn.relu(
                tf.nn.conv2d(x_reshape, filter=w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
            # 池化 2*2的窗口 黑白1张表 64个人 步长2
            # [None, 28,28,32]--->[None,14,14,32]
            x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 3. 二卷积层
        with tf.variable_scope("conv_2"):
            # 随机初始化权重 5*5的窗口*32 黑白1张表 64个人 步长1
            w_conv2 = self.weight_variables((5, 5, 32, 64))
            # 随机初始化偏置 偏置和多少个人是一样的
            b_conv2 = self.bias_variables((64,))
            # 卷积+激活
            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, filter=w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
            # [None, 14,14,32]--->[None,14,14,64]
            # 池化[None,14,14,64] -->[None,7,7,64]
            x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 4. 全连接层 [None,7,7,64]--->[None,7*7*64] * [7*7*64, 10] + [10]= [None, 10]
        with tf.variable_scope("conv_3"):
            # 初始化权重和偏置
            w_fc = self.weight_variables(shape=(7 * 7 * 64, 10))
            b_fc = self.bias_variables(shape=(10,))

            # 进行矩阵计算得出[None, 10]每个样本的10个结果

            # 矩阵相乘需要二维
            x_fc_reshape = tf.reshape(x_pool2, shape=(-1, 7 * 7 * 64))

            y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

        return x, y_true, y_predict

    def conv_fc(self):
        """
        预测函数
        :return:
        """
        x, y_true, y_predict = self.model()

        # 4. 计算所有样本的交叉熵损失 然后求平均值
        with tf.variable_scope("soft_cross"):
            # none个损失值的列表
            loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
            # 求列表损失值平均值
            loss = tf.reduce_mean(loss_list)

        # 5. 梯度下降优化求出损失
        with tf.variable_scope("optimizer"):
            train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

        # 6. 计算准确率
        with tf.variable_scope("acc"):
            equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
            # equal_list None个样本
            # [1, 0, 1, 1, 0, 1] 求平均值
            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

        # 定义一个初始化变量Op
        init_op = tf.global_variables_initializer()

        # 开启会话
        with tf.Session() as sess:
            sess.run(init_op)

            # 循环去训练
            for i in range(2000):
                # 取出真实存在的特征值和目标值
                mnist_x, mnist_y = self.mnist.train.next_batch(100)
                # 运行训练op
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                print("训练第%d步,准确率为%f,损失值为%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}),
                                                sess.run(loss, feed_dict={x: mnist_x, y_true: mnist_y})))


if __name__ == "__main__":
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    ann = Ann(mnist)
    ann.conv_fc()
    # ann.full_connected()
    # tensorboard --logdir=./summary/ann
