import tensorflow as tf
import layers
from learning_rate import learning_rate_schedule

FLAGS = tf.app.flags.FLAGS

MAX_STEPS = 64000
VAR_LIST = [0.1, 0.01, 0.001]
PIVOT_LIST = [0, 32000, 48000]
WD_FACTOR = 0.0001

def optimizer():
    lr = learning_rate_schedule(VAR_LIST, PIVOT_LIST)
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=False)

def inference(images, is_train):
    """Definition of model inference.
    Args:
      images: A batch of images to process. Shape [batch_size,32,32,3]
    """

    def shortcut(l, in_channel, out_channel, stride):
        """Shortcut for residual function.
        Args:
          l: Output of previous layer.
          in_channel: # of channels of l.
          out_channel: # of channels of each output feature.
        """
        shortcut = tf.nn.avg_pool(l, [1,stride,stride,1], [1,stride,stride,1], 'VALID')
        pad = (out_channel - in_channel)//2
        return tf.pad(shortcut, [[0,0], [0,0], [0,0], [pad, pad]])

    def residual(name, l, in_channel, out_channel, stride):
        """Residual function.
        Args:
          name: Scope name of this function.
          l: Output of previous layer.
          in_channel: # of channels of l.
          out_channel: # of channels of each output feature.
          stride: Stride of the first convolution in residual function.
        """
        with tf.variable_scope(name):
            sc = l if stride == 1 else shortcut(l, in_channel, out_channel, stride)
            l = layers.conv('conv_0', l, out_channel, stride=stride)
            l = layers.batchnorm('bn_0', l, is_train)
            l = tf.nn.relu(l)
            l = layers.conv('conv_1', l, out_channel, stride=1)
            l = layers.batchnorm('bn_1', l, is_train)
            l = tf.nn.relu(l + sc)
            return l

    # ResNet-20 inference
    with tf.variable_scope('inference'):
        l = images
        l = layers.conv('conv_init', l, 16, stride=1)

        l = residual('res_1_1', l, 16, 16, 1)
        l = residual('res_1_2', l, 16, 16, 1)
        l = residual('res_1_3', l, 16, 16, 1)

        l = residual('res_2_1', l, 16, 32, 2)
        l = residual('res_2_2', l, 32, 32, 1)
        l = residual('res_2_3', l, 32, 32, 1)

        l = residual('res_3_1', l, 32, 64, 2)
        l = residual('res_3_2', l, 64, 64, 1)
        l = residual('res_3_3', l, 64, 64, 1)

        l = layers.batchnorm('bn_0', l, is_train)
        l = tf.nn.relu(l)
        # global average pooling
        l = tf.reduce_mean(l, [1, 2])
        logits = layers.fully_connected('fc_0', l, 10)
    return logits


def loss(logits, labels):
    with tf.name_scope('loss'):
        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
        
        # regularization
        rloss = 0.0005 * tf.add_n([tf.nn.l2_loss(w) for w in weights])

        # classification
        closs = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
        total_loss = rloss + closs
    return total_loss
