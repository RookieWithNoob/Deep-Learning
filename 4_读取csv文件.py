import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def csv_read(file_list):
    """
    读取csv文件
    :param file_list: 文件路径+名字的列表
    :return: 返回读取的内容
    """
    # print(file_list)

    # 1. 构造文件队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2. 构造csv阅读器读取队列 默认以行读取 读取一行
    reader = tf.TextLineReader()

    key, value = reader.read(file_queue)

    # print(key, value)
    # 3. 对每一行内容解码
    # record_defaults:指定没一个样本的每一列类型 指定默认值 1-int默认值就是1 2-int默认值就是2 "None"-str 4.0-float
    records = [[1], [2010], [1], [2], [3], [4], ["None"], ["None"], ["None"], [1.0], [1], [1000.0], [20.0], ["None"], [0.0], [0.0], [0.0]]
    # 有几列就有几个参数接收
    # 805, 2010, 2, 3, 12, 4, NA, NA, NA, -1, 52.98, 1026.1, 8, NE, 84, 0.1, 0.1
    No,year,month,day,hour,season,PM_Jingan,PM_US_Post,PM_Xuhui,DEWP,HUMI,PRES,TEMP,cbwd,Iws,precipitation,Iprec = tf.decode_csv(value, record_defaults=records)
    # print(No, year)

    # 4. 想要读取多个数据 需要批处理
    No_batch,year_batch,month_batch,day_batch,hour_batch,season_batch,PM_Jingan_batch,PM_US_Post_batch,PM_Xuhui_batch,DEWP_batch,HUMI_batch,PRES_batch,TEMP_batch,cbwd_batch,Iws_batch,precipitation_batch,Iprec_batch = tf.train.batch([No,year,month,day,hour,season,PM_Jingan,PM_US_Post,PM_Xuhui,DEWP,HUMI,PRES,TEMP,cbwd,Iws,precipitation,Iprec], batch_size=100, num_threads=1)

    print(No_batch)
    return No_batch,year_batch,month_batch,day_batch,hour_batch,season_batch,PM_Jingan_batch,PM_US_Post_batch,PM_Xuhui_batch,DEWP_batch,HUMI_batch,PRES_batch,TEMP_batch,cbwd_batch,Iws_batch,precipitation_batch,Iprec_batch




if __name__ == "__main__":
    # 找到文件 放入列表
    file_name = os.listdir("./csv_data")
    # ['ShenyangPM20100101_20151231.csv', 'GuangzhouPM20100101_20151231.csv', 'ChengduPM20100101_20151231.csv', 'ShanghaiPM20100101_20151231.csv', 'BeijingPM20100101_20151231.csv', '911.csv']
    # print(file_name)
    file_list = [os.path.join("./csv_data", file) for file in file_name]

    No,year,month,day,hour,season,PM_Jingan,PM_US_Post,PM_Xuhui,DEWP,HUMI,PRES,TEMP,cbwd,Iws,precipitation,Iprec = csv_read(file_list=file_list)

    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        for i in range(3):
            # 打印读取的内容
            print(sess.run([No,year,month,day,hour,season,PM_Jingan,PM_US_Post,PM_Xuhui,DEWP,HUMI,PRES,TEMP,cbwd,Iws,precipitation,Iprec]))

        # 回收子线程
        coord.request_stop()
        coord.join(threads)

# 批处理大小 跟队列,数据的数量没有影响 只决定这次取多少数据出来训练 batch_size
