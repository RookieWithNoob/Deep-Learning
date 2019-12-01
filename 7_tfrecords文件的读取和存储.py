import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义cifar的数据等命令行参数
tf.app.flags.DEFINE_string("cifar_dir", "../cifar-10-binary/cifar-10-batches-bin", "文件的数据目录")
tf.app.flags.DEFINE_string("cifar_tfrecrds", "./tfrecords/cifar.tfrecords", "tfrecords文件目录")

FLAGS = tf.app.flags.FLAGS


class CifarRead():
    """
    完成读取二进制文件 写进tfrecords 读取tfrecords
    """

    def __init__(self, file_list):
        # 文件列表
        self.file_list = file_list
        print(self.file_list)
        # 定义读取的图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        # 二进制文件的每张图片的字节
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.image_bytes + self.label_bytes

    def read_and_decode(self):
        # 1. 构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)
        # 2. 构造二进制阅读器 每个样本的字节数
        reader = tf.FixedLengthRecordReader(self.bytes)
        print(self.bytes)
        key, value = reader.read(file_queue)
        print(value)
        # 3. 解码 默认一个样本 out_type输出的类型
        label_image = tf.decode_raw(value, out_type=tf.uint8)
        print(label_image)  # 1维 目标值+特征值 需要分开
        # 分割图片和标签数据 特征值 目标值
        # 从0 开始 1结束 切片
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), dtype=tf.int32)
        image = tf.cast(tf.slice(label_image, [self.label_bytes], [self.image_bytes]), dtype=tf.float32)
        print(label)
        print(image)
        # 可以对图片的特征数据进行形状的改变用于训练 [32,32,3] 只能用动态转换 静态不能跨阶
        image_reshape = tf.reshape(image, shape=(self.height, self.width, self.channel))
        print(image_reshape)
        # 4. 批处理
        label_batch, image_batch = tf.train.batch([label, image_reshape], batch_size=100, num_threads=2)
        print(label_batch, image_batch)

        return label_batch, image_batch

    def write_to_tfrecords(self, label_batch, image_batch):
        """
        将图片的特征值和目标值存进tfrecords
        :param label_batch: 特征值
        :param image_batch: 目标值
        :return: None
        """
        # 构造存储器 tfrecords存储器
        writer = tf.python_io.TFRecordWriter(path=FLAGS.cifar_tfrecrds)

        # 循环将所有样本写近去 每张图片样本都要构建example协议
        for i in range(100):
            # 取出第i个图片的特征值和目标值 tostring()
            image = image_batch[i].eval().tostring()
            label = label_batch[i].eval()[0]
            # 构造一个样本的example
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            # 写入单独的样本
            writer.write(example.SerializeToString())
        # 关闭
        writer.close()

        return None

    def read_from_tfrecords(self):
        """
        读取tfrecords文件
        :return:
        """
        # 1.构造文件队列
        print(FLAGS.cifar_tfrecrds)
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecrds])
        # 2.构造文件阅读器 string example
        reader = tf.TFRecordReader()
        # value是一个样本的序列化example
        key, value = reader.read(file_queue)
        # 3 .解析   多一步这个
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], dtype=tf.string),
            "label": tf.FixedLenFeature([], dtype=tf.int64)
        })
        print(features["image"], features["label"])
        # 4. 解码 如果读取的内容是bytes(string)需要解码 其他的int64 float32不需要
        image = tf.decode_raw(features["image"], out_type=tf.float32)
        label = tf.cast(features["label"], dtype=tf.int32)
        # 固定图片的形状 方便批处理
        print(image.shape)
        image_reshape = tf.reshape(image, shape=(self.height, self.width, self.channel))
        print(image_reshape, label)
        # 批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1)

        return image_batch, label_batch


if __name__ == "__main__":
    # 找到文件 放入列表
    file_name = os.listdir(FLAGS.cifar_dir)

    file_list = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]
    # print(file_list)

    cifar = CifarRead(file_list)
    # label_batch, image_batch = cifar.read_and_decode()

    # 读取tfrecords文件
    image_batch, label_batch = cifar.read_from_tfrecords()
    print(image_batch, label_batch)

    # 开启会话
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)
        # for i in range(3):
        # print(sess.run([label_batch, image_batch]))

        # 存进tfrecords文件
        # print("开始存储")
        # cifar.write_to_tfrecords(label_batch, image_batch)
        # print("结束")

        # 读取tensorflow文件
        print(sess.run([image_batch, label_batch]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)


# 保存的是什么格式 就用什么格式加载
# image = tf.cast(tf.slice(label_image, [self.label_bytes], [self.image_bytes]), dtype=tf.float32)
# image = tf.decode_raw(features["image"], out_type=tf.float32)