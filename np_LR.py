'''
 # Created by DahyeYoon on 5/19/16 at 11:13
 # Project Name: TensorflowTest2
'''

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt',unpack=True, dtype='float32')  # data(csv/text file) load
x_data = xy[0:-1]  # (0 1 2) ==> whole -1
y_data = xy[-1]  # end=(-1) or 3

print 'x', x_data
print 'y', y_data

w = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))

hypothesis = tf.matmul(w, x_data)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(w)