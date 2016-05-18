'''
 # Created by DahyeYoon on 5/18/16 at 16:20
 # Project Name: TensorflowTest2
'''

import tensorflow as tf

# training data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# weight, bias = random values
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# H(x)=w*x + b
hypothesis = w*x_data +b
# cost ==> sum{(H(x)-y<target value>)^2}/m : operation
cost = tf.reduce_mean(tf.square(hypothesis-y_data))

# Minimize cost
a = tf.Variable(0.1)  # Learning rate (alpha)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Variables(w, b) must be initialize before starting.
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)  # initialization run


#  Fitting the line
for step in xrange(2001):  # Repeat 2001 times
    sess.run(train)  # cost minimization run
    if step % 20 == 0:  # Print every 20 time
        print step, sess.run(cost), sess.run(w), sess.run(b)

# Result : When value of step is 600, cost is minimum.