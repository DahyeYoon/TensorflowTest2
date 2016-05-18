'''
 # Created by DahyeYoon on 5/18/16 at 19:54
 # Project Name: TensorflowTest2
'''


import tensorflow as tf
from matplotlib import pyplot as plt

# training data
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples =len(X)

# weight = random value
w = tf.placeholder(tf.float32)


# H(x)=w*x
hypothesis = tf.mul(X, w)

# cost ==> sum{(H(x)-y<target value>)^2}/m : operation
cost = tf.reduce_mean(tf.pow(hypothesis-Y, 2))/(m)


# Variables(w) must be initialize before starting.
init = tf.initialize_all_variables()

# For graphs
W_val = []
cost_val = []

# Launch the graph
sess = tf.Session()
sess.run(init)  # initialization run

for i in range(-30, 50):
    print i*0.1, sess.run(cost, feed_dict={w: i*0.1})
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={w: i*0.1}))


# Graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()

