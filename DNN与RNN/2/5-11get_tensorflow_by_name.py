import tensorflow as tf 

with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
	softmax_tensor=sess.graph.get_tensor_by_name("softmax:0")#softmax的第0个输出
	print(softmax_tensor)
	print(tf.global_variables())
	print(softmax_tensor.name)
