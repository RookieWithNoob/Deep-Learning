import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.app.flags.DEFINE_string("fruits_tfrecrds", "./tfrecords/train.tfrecords", "tfrecords文件目录")

FLAGS = tf.app.flags.FLAGS


class Fruits_Recognize():
    """验证码识别"""

    def __init__(self):
        self.fruits_dir = FLAGS.fruits_tfrecrds
        self.batch_size = 10

    def read_decode_data(self):
        """
        读取records文件
        :return: image_batch, label_batch
        """
        # 构造文件队列
        file_queue = tf.train.string_input_producer([self.fruits_dir])

        # 构造阅读器
        reader = tf.TFRecordReader()
        # 读取内容 默认一行
        key, value = reader.read(file_queue)

        # 解析
        features = tf.parse_single_example(
            value,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
        )
        # 解码

        image = tf.decode_raw(features["image"], tf.uint8)

        image = tf.reshape(image, [100, 100, 3])

        label = tf.cast(features['label'], tf.int32)
        # 批处理
        image_batch, label_batch = tf.train.batch([image, label], batch_size=self.batch_size, num_threads=1)
        # print(image_batch.shape)
        print(image_batch, label_batch)

        return image_batch, label_batch

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

    def model(self, image_batch):
        """
        进行预测结果
        :return: y_predict (100,120)
        """
        # 2. 一卷积层 卷积 激活 池化
        with tf.variable_scope("conv_1"):
            # 随机初始化权重 5*5的窗口 黑白1张表 32个人 步长1
            w_conv1 = self.weight_variables([5, 5, 3, 32])
            # 随机初始化偏置 偏置和多少个人是一样的
            b_conv1 = self.bias_variables([32])

            # 1. 卷积 5*5的窗口 黑白1张表 32个人 步长1
            # [None, 100, 100, 3]---->[None,100,100,32]
            # 卷积+激活
            x_relu1 = tf.nn.relu(
                tf.nn.conv2d(image_batch, filter=w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
            # 池化 2*2的窗口 黑白1张表 64个人 步长2
            # [None, 100,100,32]--->[None,50,50,32]
            x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            x_pool2 = tf.nn.dropout(x_pool1, keep_prob=0.5)

        # 3. 二卷积层
        with tf.variable_scope("conv_2"):
            # 随机初始化权重 5*5的窗口*32 黑白1张表 64个人 步长1
            w_conv2 = self.weight_variables([5, 5, 32, 64])
            # 随机初始化偏置 偏置和多少个人是一样的
            b_conv2 = self.bias_variables([64])
            # 卷积+激活
            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool2, filter=w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
            # [None, 50,50,32]--->[None,50,50,64]
            # 池化[None,50,50,64] -->[None,25,25,64]
            x_pool3 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            x_pool4 = tf.nn.dropout(x_pool3, keep_prob=0.5)

        # 4. 全连接层 [None,25,25,64]--->[None,25*25*64] * [25*25*64, 10] + [10]= [None, 10]
        with tf.variable_scope("conv_3"):
            # 初始化权重和偏置
            w_fc = self.weight_variables(shape=[25 * 25 * 64, 1024])
            b_fc = self.bias_variables(shape=[1024])

            # 进行矩阵计算得出[None, 10]每个样本的10个结果

            # 矩阵相乘需要二维
            print(x_pool4)
            x_fc_reshape = tf.reshape(x_pool4, shape=[-1, 25 * 25 * 64])

            y_predict1 = tf.matmul(x_fc_reshape, w_fc) + b_fc

        with tf.variable_scope("conv_4"):
            # 初始化权重和偏置
            w_fc1 = self.weight_variables(shape=[1024, 20])
            b_fc1 = self.bias_variables(shape=[20])

            # 进行矩阵计算得出[None, 10]每个样本的10个结果

            y_predict = tf.matmul(y_predict1, w_fc1) + b_fc1

        return y_predict

    def main(self):
        """
        通过输入图片特征数据 建立模型 得出预测结果
        :return:
        """
        image_batch, label_batch = self.read_decode_data()

        image_batch = tf.cast(image_batch, dtype=tf.float32)
        y_predict = self.model(image_batch)
        # 对label_batch进行one_hot编码
        y_true = tf.one_hot(label_batch, depth=20, axis=1, on_value=1.0)
        # 交叉熵损失
        with tf.variable_scope("soft_cross"):
            # none个损失值的列表
            loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
            loss = tf.reduce_mean(loss_list)

        # 5. 梯度下降优化求出损失
        with tf.variable_scope("optimizer"):
            # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

        # 6. 计算准确率
        with tf.variable_scope("acc"):
            equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
            # equal_list None个样本
            # [1, 0, 1, 1, 0, 1] 求平均值
            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

        # 定义一个初始化变量的op
        init_op = tf.global_variables_initializer()

        # 开启会话训练
        with tf.Session() as sess:
            sess.run(init_op)
            # 定义一个线程协调器
            coord = tf.train.Coordinator()

            # 开启读取文件的线程
            threads = tf.train.start_queue_runners(sess, coord=coord)

            for i in range(5000):

                sess.run(train_op)
                # print(y_true, y_predict)
                print("训练第%d步,准确率为%f,损失值为%f" % (i, sess.run(accuracy), sess.run(loss)))

            # 回收子线程
            coord.request_stop()
            coord.join(threads)
            # 卷积层 keep_prob_fifty:0.5 每个元素被保留下来的概率
            # 全连层 0.75
            # 预测的时候都为1.0


if __name__ == "__main__":
    cr = Fruits_Recognize()
    cr.main()
