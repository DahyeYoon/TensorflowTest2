'''
 # Created by DahyeYoon on 5/19/16 at 13:35
 # Project Name: TensorflowTest2
'''

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train3.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])
# x_data = xy[0:3]
# y_data = xy[3:]

print 'x', x_data
print 'y', y_data

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

w = tf.Variable(tf.zeros([3, 3]))


hypothesis = tf.nn.softmax(tf.matmul(X, w))

learning_r = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_r).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(w)

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data})