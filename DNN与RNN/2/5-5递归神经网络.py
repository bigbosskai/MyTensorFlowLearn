import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#输入的图片是28*28
n_inputs=28 #输入一行，一行有28个数据
max_time=28 #一共有28行
lstm_size=100 #隐层单元

# 每一次输入图片的一行，总共输入28次（每一行相当于一个时间点，所以要输入28次）

n_classes=10
batch_size=50
n_batch=mnist.train.num_examples // batch_size

x=tf.placeholder(tf.float32,[None,784])

y=tf.placeholder(tf.float32,[None,10])

#初始化权重
weights=tf.Variable(tf.truncated_normal( [lstm_size,n_classes], stddev=0.1))
#[100，,10]


#初始化偏执
biases=tf.Variable(tf.constant(0.1,shape=[n_classes]))
# [10]

# 定义RNN网络
def RNN(X,weights,biases):
	"""
		每一行是一个时间点
		inputs:[batch_size,max_time,n_input]
	"""
	inputs=tf.reshape(X,[-1,max_time,n_inputs])
	# 定义LSTM基本cell
	lstm_cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
	# final_state[0]是cell_state
	# final_state[1]hidden_state

	#下面相当于是一个计算
	outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
	results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
	return results
  
prediction=RNN(x,weights,biases)
#cross_entropy
cross_entropy=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal( tf.argmax(prediction,1), tf.argmax(y,1) )

accuracy=tf.reduce_mean( tf.cast( correct_prediction ,tf.float32 ))

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(6):
		for batch in range(n_batch):
			batch_xs,batch_ys=mnist.train.next_batch(batch_size)
			sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
		acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
		print("Iter "+str(epoch)+" Accuracy: "+str(acc))

"""
Iter 0 Accuracy: 0.7267
Iter 1 Accuracy: 0.8685
Iter 2 Accuracy: 0.9046
Iter 3 Accuracy: 0.9156
Iter 4 Accuracy: 0.928
Iter 5 Accuracy: 0.9354
"""