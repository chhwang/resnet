import tensorflow as tf
import numpy as np

def conv(name, l, out_channel, stride=1, kernel_size=3, padding='SAME', collections=[]):
    """Convolutional layer.

    Args:
      name:         (str) Scope name of this function
      l:            (Tensor) Output of previous layer
      out_channel:  (int) # of channels of each output feature
      stride:       (int) Stride of convolution
      kernel_size:  (int) Length of a side of convolution filter
      padding:      (str) 'SAME' to use padding, or 'VALID'
      collections:  (list) Additional collections
    """
    in_channel = l.get_shape().as_list()[3]
    with tf.variable_scope(name):
        n = kernel_size * kernel_size * out_channel
        weights = tf.get_variable('weights',
            shape=[kernel_size, kernel_size, in_channel, out_channel],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.WEIGHTS]+collections)
        return tf.nn.conv2d(l, weights, [1, stride, stride, 1], padding=padding)


def fully_connected(name, l, out_dim, collections=[]):
    """Fully connected layer.

    Args:
      name:     (str) Scope name of this function
      l:        (Tensor) Output of previous layer
      out_dim:  (int) Dimension of each output feature
    """
    with tf.variable_scope(name):
        reshape_size = 1
        for i in l.get_shape().as_list()[1:]:
            reshape_size = reshape_size * i
        l = tf.reshape(l, [-1, reshape_size])
        weights = tf.get_variable('weights',
            shape=[l.get_shape()[1], out_dim],
            dtype=tf.float32,
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.WEIGHTS]+collections)
        biases = tf.get_variable('biases',
            shape=[out_dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.BIASES]+collections)
        return tf.nn.xw_plus_b(l, weights, biases)


def batchnorm(name, l, is_train):
    """Batch normalization layer.

    Args:
      name:     (str) Scope name of this function
      l:        (Tensor) Output of previous layer
      is_train: (Tensor) Whether to train or not
    """
    in_channel = l.get_shape().as_list()[3]
    with tf.variable_scope(name):
        beta = tf.get_variable('beta',
            shape=[in_channel],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma',
            shape=[in_channel],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))
        batch_mean, batch_var = tf.nn.moments(l, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        tf.add_to_collection('ema_apply_ops', ema.apply([batch_mean, batch_var]))
        ema_mean, ema_var = (ema.average(batch_mean), ema.average(batch_var))
        mean, var = tf.cond(is_train,
                            lambda: (batch_mean, batch_var),
                            lambda: (ema_mean, ema_var))
        return tf.nn.batch_normalization(l, mean, var, beta, gamma, 1e-3)
