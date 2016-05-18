'''
 # Created by DahyeYoon on 5/18/16 at 19:42
 # Project Name: TensorflowTest2
'''


import tensorflow as tf

# training data
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# weight = random value
w = tf.Variable(tf.random_uniform([1], -10, 10))


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# H(x)=w*x
hypothesis = w*X
# cost ==> sum{(H(x)-y<target value>)^2}/m : operation
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# Minimize using gradient descent
descent = w- tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(w, X)-Y), X)))
update = w.assign(descent)  # operation. This cannot running


# Variables(w) must be initialize before starting.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)  # run initialization


#  Fitting the line
for step in xrange(100):  # Repeat 100 times
    sess.run(update, feed_dict={X: x_data, Y: y_data})  # run update
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(w)  # minimize cost
