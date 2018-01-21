import tensorflow as tf
"""
case 1.
input=tf.Variable(tf.random_normal([1,3,3,5]))
filter=tf.Variable(tf.random_normal([1,1,5,1]))
op=tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='VALID')
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    res=(sess.run(op))
    print(res)
"""
"""
case 2.
input=tf.Variable( tf.random_normal([1,5,5,5]))
filter=tf.Variable( tf.random_normal([3,3,5,1]))
op=tf.nn.conv2d( input, filter, strides=[1,1,1,1],padding='VALID')
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    res=(sess.run( op ))
    print(res.shape)
"""

