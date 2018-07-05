import os
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 解决Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2  问题
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 导入数据，用于测试
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 读取并重建模型
sess = tf.Session()
saver = tf.train.import_meta_graph('./save/model.meta')
saver.restore(sess, './save/model')

graph = tf.get_default_graph()
input_x = graph.get_operation_by_name('input_x').outputs[0]
keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
y = tf.get_collection("y_conv")[0]

# 获取一个测试数据
data = mnist.validation.next_batch(1)


# 将测试数据的 data[0] image信息喂给模型，并获取输出结果
res = sess.run(y, feed_dict={input_x: data[0], keep_prob: 1.0})

# 将测试数据的 data[1] lable信息取出，并转化为list
list_a = data[1][0, :].tolist()
# 查询list的最大元素的位置， 即对应手写体的数字
max_index = list_a.index(max(list_a))
# 打印实际的数据
print("lable ", list_a)
print("lable ", max_index)

# 将输出数据转化为list
list_b = res[0, :].tolist()
# 查询list中最大元素的位置， 即预测的手写体的数字
max_index_b = list_b.index(max(list_b))
# 打印预测结果
print("res ", list_b)
print("res ", max_index_b)

