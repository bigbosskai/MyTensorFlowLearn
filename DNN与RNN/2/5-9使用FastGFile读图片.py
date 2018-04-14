import tensorflow as tf 
image_jpg=tf.gfile.FastGFile("1.jpg","rb").read()
image_png=tf.gfile.FastGFile("2.png","rb").read()

with tf.Session() as sess:
	print(image_jpg)
	image_jpg=tf.image.decode_jpeg(image_jpg) #图像解码
	print(sess.run(image_jpg))#打印解码后的图像（即为一个三维矩阵[w,h,3]）
	mage_jpg = tf.image.convert_image_dtype(image_jpg,dtype=tf.uint8) #改变图像数据类型  
	print(sess.run(mage_jpg))

	print(type(image_jpg))
	print(mage_jpg.get_shape())

	image_png = tf.image.decode_png(image_png) 
    print(sess.run(image_jpg))
    image_png = tf.image.convert_image_dtype(image_png,dtype=tf.uint8) 