import tensorflow as tf
import numpy as np
#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
print(x_data.shape)
#create tensorflow structure start
Weights = tf.Variable( tf.random_normal([1]) )
biases = tf.Variable( tf.zeros([1]) )

y=Weights*x_data + biases
loss=tf.reduce_mean( tf.square(y-y_data) )
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run( init )
    for step in range(1,201):
        sess.run(train)
        if step%20==0 or step==1:
            print(step,sess.run(Weights),sess.run(biases))



import numpy as np
arr = np.array([1,2,3,4,5])
print(arr)
print(2*arr)
print(2*arr+1)
"""
outputs:
[1 2 3 4 5]
[ 2  4  6  8 10]
[ 3  5  7  9 11]
"""