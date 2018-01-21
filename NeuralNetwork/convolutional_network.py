import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("",one_hot=True)
#Parameters
learning_rate=0.001
training_iters=2000
batch_size=128
display_step=100
#Network Parameters
n_input=784
n_classes=10
dropout=0.75
#tf Graph input

x=tf.placeholder(tf.float32,shape=[None,n_input])
y=tf.placeholder(tf.float32,shape=[None,n_classes])
keep_prob=tf.placeholder(tf.float32)#dropout (keep probability)

#create some wrappers for simplicity

def conv2d(x,W,b,strides=1):
    """Conv2d wrapper, with bias and relu activation"""
    x=tf.nn.conv2d( x, W , strides=[1,strides,strides,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    """MaxPool2d wrapper"""
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

# Store layers weight & bias

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}


biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def conv_net(x,weights,biases,dropout):
    """Reshape input picture"""
    x=tf.reshape(x,shape=[-1,28,28,1])
    #Convolution Layer
    conv1=conv2d(x,weights['wc1'],biases['bc1'])#[?,28,28,32]
    conv1=maxpool2d(conv1,k=2)#[?,14,14,32]
    #conv1 [?,14,14,32]
    conv2=conv2d( conv1, weights['wc2'],biases['bc2'])
    #[?,14,14,64]
    conv2=maxpool2d(conv2,k=2)
    #[?,7,7,64]

    #Reshape conv2 ouput to fit fully connected layer input
    fc1=tf.reshape(conv2,[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1=tf.add( tf.matmul( fc1, weights['wd1']),biases['bd1'])
    fc1=tf.nn.relu(fc1)

    fc1=tf.nn.dropout(fc1, dropout)
    out=tf.add(tf.matmul( fc1, weights['out']),biases['out'])

    return out

pred=conv_net(x,weights,biases,keep_prob)
cost=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred) )
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluate model
correct_pred= tf.equal(tf.argmax(pred ,1) , tf.argmax(y,1))
accuracy=tf.reduce_mean( tf.cast(correct_pred,tf.float32))
#Initializeing the variables
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run( init )
    for step in range(1,training_iters+1):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
        if step%display_step==0 or step==1:
            loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y
                                                         ,keep_prob:1.0})
            print("Iter"+str(step)+" Minibatch Loss="+"{:.6f}".format(loss)
                  +"Train Accuracy"+"{:.5f}".format(acc)
                  )
    print("Optimization Finished!")
    print("Testing on test.iamges")
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:1000],
                                      y: mnist.test.labels[:1000],
                                      keep_prob: 1.}))
