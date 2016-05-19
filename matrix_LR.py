'''
 # Created by DahyeYoon on 5/19/16 at 10:46
 # Project Name: TensorflowTest2
'''

import tensorflow as tf
# [b w1 w2] * [1 x1 x2]' = [b*1 + w1*x1 + w2*x2]
x_data = [[0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]
y_data = [1, 2, 3, 4, 5]

w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))  # w is 2-dimensional array
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = tf.matmul(w, x_data) + b  # matmul = matrix multiplication
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(w), sess.run(b)


