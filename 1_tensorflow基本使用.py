import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一张图  上下文管理器
# op:只要使用tensorflow的api接口定义的函数都是op 相当于是载体
# tensor : 就指代的是数据 被载体
g = tf.Graph()

print("新的图:", g)
with g.as_default():
    c = tf.constant(11.0)
    print("c的图:", c.graph)

# 实现一个加法运算
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a, b)
print("sum1:", sum1)

# 有重载的机制 默认会给运算符重载成op类型
var1 = 2.0
sum2 = a + var1
print("重载sum2:", sum2)
# 默认的这张图　相当于是给程序分配一段内存
graph = tf.get_default_graph()
print("默认图:", graph)

# 训练模型
# 实时的提供数据去进行训练

# placeholder是一个占位符　feed_dict是一个字典
plt = tf.placeholder(tf.float32, (None, 3))
print("plt:", plt)

# 只能运行一个图结构 可以在会话当中指定图去运行graph=
# 只要有会话的上下文环境　就可以使用方便的eval()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)
                ) as sess:
    # 不是op tensor不能运行
    print(sess.run([a, b, sum1, sum2]))
    print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6], [11, 22, 33]]}))
    # print(sum1.eval())
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)


# tensorflow打印出来的形状表示:
# 0维:()单纯一个数字 1维:(5) 一个列表 2维:(5, 6) 3维: (2,3,4) 0阶　1阶 2阶 3阶 ....

# 形状的概念
# 静态形状和动态形状　静态修改原形状　动态创建新的张量

plt = tf.placeholder(tf.float32, [None, 2])
print(plt)

# 对于静态形状来说　一旦张量形状固定了　就不能再次设置静态形状 不能跨阶数(维度)修改 1维到2维等等 1D->1D
plt.set_shape((4, 2))
print(plt)
# 不能再次修改
# plt.set_shape((4, 2))

# 动态形状可以去创建新的张量 注意元素数量要匹配　shape[0]*shape[1] 1D->2D
plt_reshape = tf.reshape(plt, (2, 4))
print(plt_reshape)
