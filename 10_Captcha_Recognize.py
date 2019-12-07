import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.app.flags.DEFINE_string("captcha_dir", "./tfrecords/captcha.tfrecords", "验证码tfrecords文件路径")

FLAGS = tf.app.flags.FLAGS


class CaptchaRecognize():
    """验证码识别"""

    def __init__(self):
        self.captcha_dir = FLAGS.captcha_dir
        self.batch_size = 100

    def read_decode_data(self):
        """
        读取records文件
        :return: image_batch, label_batch
        """
        # 构造文件队列
        file_queue = tf.train.string_input_producer([self.captcha_dir])

        # 构造阅读器
        reader = tf.TFRecordReader()
        # 读取内容 默认一行
        key, value = reader.read(file_queue)

        # 解析
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string)
        })

        # 解码
        # 1.先解码图片的特征值
        image = tf.decode_raw(features["image"], tf.uint8)
        # 2.解码图片的目标值
        label = tf.decode_raw(features["label"], tf.uint8)
        # print(image, label)

        # 设置形状
        image_reshape = tf.reshape(image, shape=(20, 80, 3))
        label_reshape = tf.reshape(label, shape=(4,))
        # print(image_reshape, label_reshape)

        # 批处理 每次读取的样本数量 100
        # 也就是每次训练时候的样本数
        image_batch, label_batch = tf.train.batch([image_reshape, label_reshape], batch_size=self.batch_size,
                                                  num_threads=1, capacity=100)
        print(image_batch, label_batch)

        return image_batch, label_batch


    def model(self, image_batch):
        """
        进行预测结果
        :param image_batch: 图片的特征值
        :return: y_predict (100,104)
        """
        with tf.variable_scope("model"):
            # 随机初始化权重和偏置
            weight = tf.Variable(tf.random_normal(shape=(20*80*3, 4*26), mean=0.0, stddev=1.0))
            bias = tf.Variable(tf.constant(0.0, shape=(104, )))
            # 预测输出结果
            image_batch = tf.cast(tf.reshape(image_batch, shape=(-1, 20*80*3)), dtype=tf.float32)
            y_predict = tf.matmul(image_batch, weight) + bias
            return y_predict

    def main(self):
        """
        通过输入图片特征数据 建立模型 得出预测结果
        :return:
        """
        image_batch, label_batch = self.read_decode_data()
        # 一层 不用卷积层 直接全连接层
        # y = (100, 20, 80, 3) * (20*80*30, 104) = (100,104)
        # 权重为(20*80*3, 104) bias 为(104, )
        y_predict = self.model(image_batch)
        # 对label_batch进行one_hot编码
        y_true_acc = tf.one_hot(label_batch, depth=26, axis=2, on_value=1.0)
        print("y_true_acc",y_true_acc)
        y_true  = tf.reshape(y_true_acc, shape=(100,104))
        print("y_true", y_true)
        y_predict_acc = tf.reshape(y_predict, shape=(100, 4, 26))
        print("y_predict", y_predict)
        print("y_predict_acc", y_predict_acc)

        # 交叉熵损失
        with tf.variable_scope("soft_cross"):
            # none个损失值的列表
            loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
            # y_predict(100, 104)
            # y_true(100,4)
            # [[13, 20, 2, 16], [1, 2, 3, 4],]
            # 求列表损失值平均值
            loss = tf.reduce_mean(loss_list)

        # 5. 梯度下降优化求出损失
        with tf.variable_scope("optimizer"):
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            # train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

        # 6. 计算准确率
        with tf.variable_scope("acc"):
            equal_list = tf.equal(tf.argmax(y_true_acc, 2), tf.argmax(y_predict_acc, 2))
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
                print("训练第%d步,准确率为%f,损失值为%f" % (i, sess.run(accuracy),sess.run(loss)))

            # 回收子线程
            coord.request_stop()
            coord.join(threads)
            # 卷积层 keep_prob_fifty:0.5 每个元素被保留下来的概率
            # 全连层 0.75
            # 预测的时候都为1.0


if __name__ == "__main__":
    cr = CaptchaRecognize()
    cr.main()
