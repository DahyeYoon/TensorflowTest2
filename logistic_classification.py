'''
 # Created by DahyeYoon on 5/19/16 at 11:34
 # Project Name: TensorflowTest2
'''

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train2.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(w, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))  # sigmoid

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})  # run(train)==> w update
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(w)