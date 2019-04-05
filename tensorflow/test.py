import tensorflow as tf
import numpy as np

# g1=tf.Graph()
# with g1.as_default():
#     v=tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer)
#
# g2=tf.Graph()
# with g2.as_default():
#     v=tf.get_variable("v",shape=[1],initializer=tf.ones_initializer)
#
# with tf.Session(graph=g1) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable("v")))
#
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable("v")))

# a = tf.constant([1, 2], name="a", dtype=tf.float32)
# b = tf.constant([2.0, 3.0], name="b")
# result = tf.add(a, b, name="add")
# print(result)
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
# result = v1 + v2
#
# saver = tf.train.Saver([v1])
#
# with tf.Session() as sess:
#     saver.restore(sess, "/path/to/model/model.ckpt")
#     print(sess.run(result))
print(np.random.randint(100))