import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def pic_read(file_list):
    """
    读取图片并转换成张量
    :param file_list: 文件路径+ 名字的列表
    :return: 每张图片的张量
    """

    # 1. 构造文件队列
    file_queue = tf.train.string_input_producer(file_list)
    # 2.构造阅读器去读取内容 默认读取一张图片
    reader = tf.WholeFileReader()
    key , value = reader.read(file_queue)
    print(value)
    # 3.解码 jpg格式
    image = tf.image.decode_jpeg(value)
    print(image)
    # 处理图片的大小 统一大小
    image_resize = tf.image.resize_images(image, size=[500, 500])
    print(image_resize)
    # 注意 一定要把样本的形状固定(通道数 灰色还是彩色)
    image_resize.set_shape((500,500,3))
    print(image_resize)
    # 4.进行批处理 要求所有数据形状必须定义
    image_batch = tf.train.batch([image_resize], batch_size=5, num_threads=1)
    print(image_batch)

    return image_batch


if __name__ == "__main__":
    # 找到文件 放入列表
    file_name = os.listdir("./image_data")

    file_list = [os.path.join("./image_data", file) for file in file_name]
    # print(file_list)

    image_batch = pic_read(file_list)

    # 开启会话
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        print(sess.run([image_batch]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)