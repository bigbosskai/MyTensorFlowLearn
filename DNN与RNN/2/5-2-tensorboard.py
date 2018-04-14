
# coding: utf-8

# In[2]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean=tf.reduce_mean(var)
		tf.summary.scalar('mean',mean)#平均值
		with tf.name_scope('stddev'):
			stddev=tf.sqrt( tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('stddev',stddev)#标准差
		tf.summary.scalar('max',tf.reduce_max(var))#最大值
		tf.summary.scalar('min',tf.reduce_min(var))#最小值
		tf.summary.histogram('histogram',var)#直方图


#定义两个placeholder
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32,[None,784],name='x-input')
	y = tf.placeholder(tf.float32,[None,10],name='y-input')

#创建一个简单的神经网络
with tf.name_scope('layer'):
	with tf.name_scope('weights'):
		W = tf.Variable(tf.zeros([784,10]),name='W')
		variable_summaries(W)
	with tf.name_scope('biases'):
		b = tf.Variable(tf.zeros([10]),name='b')
		variable_summaries(b)
	with tf.name_scope('xW_plus_b'):
		xW_plus_b=tf.matmul(x,W)+b
	with tf.name_scope('softmax'):
		prediction = tf.nn.softmax(xW_plus_b)

#二次代价函数
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(y-prediction))
	#variable_summaries(loss) loss只有一个值
	tf.summary.scalar('loss',loss)
#使用梯度下降法
with tf.name_scope('train_step'):
	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
with tf.name_scope('acc'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y-prediction,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		tf.summary.scalar('accuracy',accuracy)

#合并所有的summary
merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs/',sess.graph)
    print(prediction.get_shape())
    for epoch in range(10):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        
        writer.add_summary(summary,epoch)

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))






