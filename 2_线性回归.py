import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义命令行参数
# 1.首先确定你想哪些参数在命令行输入
# 2.程序当中获取命令行输入的参数

# 参数: 名字 默认值 说明
tf.app.flags.DEFINE_integer("max_step", 500, "模型训练的步数")
tf.app.flags.DEFINE_string("model_dir", "./ckpt/model", "模型文件的保存和加载路径")

# 定义获取命令行参数名字
FLAGS = tf.app.flags.FLAGS
# python3 3_线性回归.py --max_step=500 --model_dir="./ckpt/model"

def myregression():
    """
    自己实现一个线性回归预测
    :return: None
    """
    # tf.variable_scope 相当于def

    with tf.variable_scope("data"):
        # 1. 准备数据 x 特征值[100, 1] y 目标值[100, 1]
        x = tf.random_normal((100000, 1), mean=1.75, stddev=0.5, name="x_data")

        # (100, 1) * (1, 1) = (100, 1)
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2. 建立线性回归模型 1个特征 就有1个权重 加上一个偏置 y = kx + b 初始化先随机给权重值和偏置 然后在当前状态下进行梯度下降优化

        # 一定要用变量定义才能优化 更新参数/权重
        # trainable=True:指定这个变量能否跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal((1, 1), mean=0.0, stddev=1.0, name="w"))

        bias = tf.Variable(0.0, name="b")

        # 预测
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 建立损失函数 均方误差
        # y_true - y_predict 二阶的(100, 1)原来的形状
        loss = tf.reduce_mean(tf.square(y_true - y_predict))  # 损失值

        # Tensor("Square_1:0", shape=(100, 1), dtype=float32)
        # print(tf.square(y_true - y_predict))
        # print(loss)

    with tf.variable_scope("optimizer"):
        # 4. 梯度下降优化损失  learning_rate:  minimize 最小损失 0.01 0.001 0.0001
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 5. 定义一个所有变量初始化的op
    init_op = tf.global_variables_initializer()

    # 6. 收集tensor (变量) 损失值 权重 准确率 偏置
    # 单值
    tf.summary.scalar(name="losses", tensor=loss)
    tf.summary.scalar(name="pianzhi", tensor=bias)
    # 多维 1 2 3
    tf.summary.histogram(name="weights", values=weight)

    # 7. 定义合并tensor 的op
    merged = tf.summary.merge_all()

    # 8. 定义一个保存模型的实例
    saver = tf.train.Saver()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print("随机初始化的权重参数为:{}  偏置为:{}".format(sess.run(weight), bias.eval()))
        print(type(sess.run(weight)))

        # 建立事件文件
        filewriter = tf.summary.FileWriter("./summary/test", graph=sess.graph)

        # 在训练之前加载模型 覆盖参数 从上次训练的参数结果开始   判断文件是否存在
        if os.path.exists("./ckpt/checkpoint"):
            # 存在则加载模型
            saver.restore(sess, FLAGS.model_dir)

        # 循环训练 运行优化 步数
        for i in range(FLAGS.max_step):

            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)
            filewriter.add_summary(summary, i)

            print("第%d次训练的权重参数为:%.3f  偏置为:%f 均方误差:%f " % (i, weight.eval(), bias.eval(), loss.eval()))

            # 每200次保存一次模型
            # if i % 200 == 0:
            #     saver.save(sess, "./ckpt/model")

        # 保存模型
        saver.save(sess, FLAGS.model_dir)


    return None


if __name__ == "__main__":
    myregression()
    # 1. 训练参数问题 trainable
    # 学习率和步数的问题
    # 2. 添加权重参数 损失值 准确率等等在tensorboard观察的情况
    # 2.1 收集变量
    # 2.2 合并变量写入事件文件

