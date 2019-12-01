import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 模拟一下    同步  先处理数据 然后才能取数据训练

# # 1.首先定义队列 capacity队列大小
# Q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
#
# # 放入一些数据
# enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#
# # 2. 定义一些处理数据逻辑 取数据的过程
# out_q = Q.dequeue() # op
#
# data = out_q + 1 # 重载
#
# en_q = Q.enqueue(data) # op
#
# with tf.Session() as sess:
#     # 初始化队列
#     sess.run(enq_many)
#
#     # 处理数据
#     for i in range(100):
#         sess.run(en_q)
#
#     # 训练数据
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))



# 模拟    异步  子线程处理数据 主线程负责读取

# 定义一个队列
Q = tf.FIFOQueue(capacity=10000, dtypes=tf.float32)

# 定义子线程要做的事情 读取数据/ 循环值+1 放入队列中
var = tf.Variable(0.0)

# 实现一个自增 tf.assign_add
data = tf.assign_add(var, tf.constant(1.0))

en_q = Q.enqueue(data)

# 定义个队列管理器op 就是创建线程 指定多少个子线程要做的事情
# 子线程该干什么事情 enqueue_ops列表 * n n个线程去做
qr = tf.train.QueueRunner(queue=Q, enqueue_ops=[en_q]*2)

# 5. 定义一个所有变量初始化的op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # 开启线程管理器
    coord = tf.train.Coordinator()

    # 真正开启子线程
    threads = qr.create_threads(sess, coord=coord, start=True)

    # 主线程 不断的读取数据训练
    for i in range(300):
        print(sess.run(Q.dequeue()))

    # 回收子线程
    coord.request_stop() # 强行
    # coord.should_stop() # 询问 检查是否要求停止

    coord.join(threads)















