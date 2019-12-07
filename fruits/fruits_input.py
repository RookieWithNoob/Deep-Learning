import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""--------------------------------------------------------------
用训练数据生成TFRecord文件
write_image_data – 将图像原始数据连同图像的正确标签保存在TFRecodes文件中
标签被编码成一个整数，是训练文件类的编号，这里一个文件夹代表一类水果
图像被保存为一个字节序列
------------------------------------------------------------------"""

# 定义cifar的数据等命令行参数
tf.app.flags.DEFINE_string("fruits_tfrecrds", "./tfrecords/train.tfrecords", "tfrecords文件目录")

FLAGS = tf.app.flags.FLAGS

class Write_Image_Data():
    """生成tfrecords文件"""

    def __init__(self):
        self.dir_name = os.getcwd() + "/data/fruits-360/Training"

    def read_then_write(self):
        # 构造存储器 tfrecords存储器
        writer = tf.python_io.TFRecordWriter(path=FLAGS.fruits_tfrecrds)
        # 构造字符串索引 {0:'Tamarillo', 1:'Apple Golden 1'}
        index_fruits = dict(enumerate(os.listdir(self.dir_name)))
        # 字符串反转 {'Tamarillo':0, 'Apple Golden 1':1}
        fruits_index = dict(zip(index_fruits.values(), index_fruits.keys()))
        # 图片列表
        image_list = []

        for fruits_name in os.listdir(self.dir_name):
            class_path = os.path.join(self.dir_name, fruits_name)
            for image_name in os.listdir(class_path):
                image = os.path.join(class_path, image_name)
                image_list.append(image)
                # print(image)
                # 读取图片
                # image_value = tf.read_file(image)
                # 以这种方式读取图片效率更高
                with  tf.gfile.FastGFile(image, 'rb') as f:
                    image_value = f.read()
                    # 解码
                    image = tf.image.decode_jpeg(image_value)
                    # 注意 一定要把样本的形状固定(通道数 灰色还是彩色)
                    image.set_shape((100, 100, 3))
                    # print(image, fruits_name, fruits_index[fruits_name])
                    # 图片特征转换成字符串形式
                    image_string = image.eval().tostring()
                    # 构造一个样本的example
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[fruits_index[fruits_name
                                                                                       ]]))
                    }
                        )
                            )
                    # 写入单独的样本
                    writer.write(example.SerializeToString())
                    print("来自{} -- {}写入完成".format(fruits_name, image_name))
        # 关闭
        writer.close()
        print(fruits_index)
        image_count = len(image_list)

        return index_fruits, image_count


if __name__ == "__main__":
    wid = Write_Image_Data()

    # 开启会话
    with tf.Session() as sess:
        index_fruits, image_count = wid.read_then_write()
        print(index_fruits)
        print(image_count)
        # tf.one_hot([22], depth=20, axis=1, on_value=1.0).eval()


