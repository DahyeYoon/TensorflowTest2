'''
 # Created by DahyeYoon on 5/18/16 at 15:44
 # Project Name: TensorflowTest2
'''


import tensorflow as tf

hello=tf.constant('Hellooooo')
print hello

sess = tf.Session()
print sess.run(hello)
