import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 获取训练数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义占位符 输入数据x
x = tf.placeholder(tf.float32, [None, 784])
# 定义占位符 y_   为正确的值
y_ = tf.placeholder("float", [None, 10])

# 定义常量 w b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义训练模型  y = softmax( W * x + b)    y是预测值
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 计算交叉熵  cross_entropy  为对 y_ * log y 求和然后取反_
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 反向传播  0.01为学习率   用梯度下降算法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

tf.test.TestCase()

# 在session里面启动模型并初始化
sess = tf.InteractiveSession()

# 初始化全局变量
init = tf.global_variables_initializer()
sess.run(init)

# 使用训练数据集循环训练模型
for i in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 打印 W 这个可以通过debug来查看全部数据
    print(sess.run(W))
    # 打印 W 的第二列
    print(sess.run(W)[:, 1])
    # 打印 b
    print(sess.run(b))

# 对模型结果进行评估,得到的是一组布尔值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 使用测试数据集来对模型进行评估
result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

# 数据评估值
print(result)
