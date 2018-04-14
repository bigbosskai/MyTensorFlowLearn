
# coding: utf-8

# In[2]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)





#二次代价函数
#loss = tf.reduce_mean(tf.square(y-prediction))
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
loss=-tf.reduce_mean(y*tf.log(prediction))




#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step1= tf.train.AdamOptimizer(1e-2).minimize(loss)#学习率设置的比较小

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    print(prediction.get_shape())
    for epoch in range(121):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:



"""
Iter 3,Testing Accuracy 0.905
Iter 4,Testing Accuracy 0.9085
Iter 5,Testing Accuracy 0.9097
Iter 6,Testing Accuracy 0.9112
Iter 7,Testing Accuracy 0.913
Iter 8,Testing Accuracy 0.9143
Iter 9,Testing Accuracy 0.9151
Iter 10,Testing Accuracy 0.9174
Iter 11,Testing Accuracy 0.9182
Iter 12,Testing Accuracy 0.918
Iter 13,Testing Accuracy 0.9192
Iter 14,Testing Accuracy 0.9201
Iter 15,Testing Accuracy 0.9206
Iter 16,Testing Accuracy 0.9206
Iter 17,Testing Accuracy 0.9211
Iter 18,Testing Accuracy 0.9207
Iter 19,Testing Accuracy 0.9209
Iter 20,Testing Accuracy 0.9225
"""