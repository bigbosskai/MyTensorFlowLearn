import tensorflow as tf
x1=tf.constant(1.0 , shape=[1,3,3,1])
kernel=tf.constant(1.0,shape=[3,3,3,1])
#[filter_height * filter_width * in_channels, output_channels]
x2=tf.constant(1.0,shape=[1,6,6,3])
x3=tf.constant(1.0,shape=[1,5,5,3])
#对x3做卷积操作
y3=tf.nn.conv2d( x3, kernel , strides=[1,2,2,1],padding="SAME")
init=tf.global_variables_initializer()
with tf.Session() as sess:
    #print(sess.run(x1))
    print(sess.run(y3))
    print(y3)

"""


    out_height = ceil(float(in_height) / float(strides[1]))

    out_width = ceil(float(in_width) / float(strides[2]))

    For the VALID padding, the output height and width are computed as:

    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))

    out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))



"""


"""
conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    Computes a 2-D convolution given 4-D `input` and `filter` tensors.
    
    Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
    and a filter / kernel tensor of shape
    `[filter_height, filter_width, in_channels, out_channels]`, this op
    performs the following:
    
    1. Flattens the filter to a 2-D matrix with shape
       `[filter_height * filter_width * in_channels, output_channels]`.
    2. Extracts image patches from the input tensor to form a *virtual*
       tensor of shape `[batch, out_height, out_width,
       filter_height * filter_width * in_channels]`.
    3. For each patch, right-multiplies the filter matrix and the image patch
       vector.
    
    In detail, with the default NHWC format,
    
        output[b, i, j, k] =
            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                            filter[di, dj, q, k]
    
    Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
    
    Args:
      input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      filter: A `Tensor`. Must have the same type as `input`.
      strides: A list of `ints`.
        1-D of length 4.  The stride of the sliding window for each dimension
        of `input`. Must be in the same order as the dimension specified with format.
      padding: A `string` from: `"SAME", "VALID"`.
        The type of padding algorithm to use.
      use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
      data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        Specify the data format of the input and output data. With the
        default format "NHWC", the data is stored in the order of:
            [batch, in_height, in_width, in_channels].
        Alternatively, the format could be "NCHW", the data storage order of:
            [batch, in_channels, in_height, in_width].
      name: A name for the operation (optional).
    
    Returns:
      A `Tensor`. Has the same type as `input`.
"""