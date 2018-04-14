import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples // batch_size
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
lr=tf.Variable(0.001,dtype=tf.float32)

W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1=tf.Variable(tf.zeros([500])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

W2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2=tf.Variable(tf.zeros([300])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)
scewl=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)
loss = tf.reduce_mean(scewl)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
init=tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    print(scewl.get_shape())
    for epoch in range(51):
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print ("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc) + ", Learning Rate= " + str(learning_rate))


    print("end")


"""
Iter 0, Testing Accuracy= 0.9447, Learning Rate= 0.001
Iter 1, Testing Accuracy= 0.9608, Learning Rate= 0.00095
Iter 2, Testing Accuracy= 0.9692, Learning Rate= 0.0009025
Iter 3, Testing Accuracy= 0.973, Learning Rate= 0.000857375
Iter 4, Testing Accuracy= 0.9732, Learning Rate= 0.000814506
Iter 5, Testing Accuracy= 0.9759, Learning Rate= 0.000773781
Iter 6, Testing Accuracy= 0.9758, Learning Rate= 0.000735092
Iter 7, Testing Accuracy= 0.9764, Learning Rate= 0.000698337
Iter 8, Testing Accuracy= 0.975, Learning Rate= 0.00066342
Iter 9, Testing Accuracy= 0.9769, Learning Rate= 0.000630249
Iter 10, Testing Accuracy= 0.977, Learning Rate= 0.000598737
Iter 11, Testing Accuracy= 0.9787, Learning Rate= 0.0005688
Iter 12, Testing Accuracy= 0.9787, Learning Rate= 0.00054036
Iter 13, Testing Accuracy= 0.9803, Learning Rate= 0.000513342
Iter 14, Testing Accuracy= 0.9792, Learning Rate= 0.000487675
Iter 15, Testing Accuracy= 0.9818, Learning Rate= 0.000463291
Iter 16, Testing Accuracy= 0.9805, Learning Rate= 0.000440127
Iter 17, Testing Accuracy= 0.9795, Learning Rate= 0.00041812
Iter 18, Testing Accuracy= 0.9829, Learning Rate= 0.000397214
Iter 19, Testing Accuracy= 0.978, Learning Rate= 0.000377354
Iter 20, Testing Accuracy= 0.9798, Learning Rate= 0.000358486

"""
"""
keep_prob:0.5
效果没有上面的好
"""