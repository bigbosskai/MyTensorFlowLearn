import tensorflow as tf 

with tf.gfile.FastGFile("inception_model/classify_image_graph_def.pb", 'rb') as f:
	print(f.read())