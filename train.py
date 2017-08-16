from __future__ import print_function
from datetime import datetime
import tensorflow as tf
import numpy as np
import cifar
import time, sys, os
import resnet

tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch.')
tf.app.flags.DEFINE_integer('gpu', 0, 'GPU to use')
tf.app.flags.DEFINE_boolean('test', True, 'Run test if True else run train')

FLAGS = tf.app.flags.FLAGS

# Set GPU to use. Only one GPU supported.
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

def main(argv=None):
    """Main function.
    """
    # prepare for checkpoint
    ckpt_dir = './ckpt'
    ckpt_path = ckpt_dir + '/train_result.ckpt'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # number of trained mini-batches
    global_step = tf.train.create_global_step()

    # True to train model, False to test model
    is_train = tf.placeholder(tf.bool)

    with tf.device('/cpu:0'):
        # get CIFAR-10 data
        train_images, train_labels = cifar.inputs('./dataset/cifar', FLAGS.batch_size, test=False)
        test_images, test_labels = cifar.inputs('./dataset/cifar', FLAGS.batch_size, test=True)
        images, labels = tf.cond(is_train,
                                 lambda: (train_images, train_labels),
                                 lambda: (test_images, test_labels))
    with tf.device('/gpu:0'):
        # ResNet inference
        logits = resnet.inference(images, is_train)
        loss = resnet.loss(logits, labels)

        # back-prop
        train = [resnet.optimizer().minimize(loss, global_step=global_step),
            tf.get_collection('ema_apply_ops')]

    with tf.device('/cpu:0'):
        # error rate
        prediction = tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)
        error = 100. * tf.reduce_mean(tf.cast(tf.not_equal(prediction,labels), tf.float32))

    # period of evaluation
    eval_period = cifar.TRAIN_SIZE // FLAGS.batch_size

    # record the time when training starts
    start_time = time.time()
    curr_time = start_time
    epoch = 0
    max_test_step = cifar.TEST_SIZE // FLAGS.batch_size


    # create a local session to run training
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # ckpt saver
    saver = tf.train.Saver()

    sys.stdout.write('Initializing ... ')
    sys.stdout.flush()

    # initialize model
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    tf.train.start_queue_runners(sess=sess)

    sys.stdout.write('Done\n')
    sys.stdout.flush()

    # loop through training steps
    for step in xrange(resnet.MAX_STEPS):
        # run training
        _, gstep, l = sess.run([train, global_step, loss], feed_dict={is_train: True})

        # periodic evaluation
        if step % eval_period == 0:
            elapsed_time = time.time() - curr_time

            # run evaluation
            errors = []
            for test_step in xrange(max_test_step):
                errors.append(sess.run(error, feed_dict={is_train: False}))
            e = sum(errors)/len(errors)

            # print progress
            sys.stdout.write('[%s(+%.1f min)] '
                             'Epoch %d; Loss %.6f; Test Error %.2f %% (%.1f ms/step)\n' %
                             (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              float(time.time()-start_time)/60.,
                              epoch, l, e, 1000*elapsed_time/eval_period))
            sys.stdout.flush()

            # shuffle train data
            epoch += 1
            curr_time = time.time()
            saver.save(sess, ckpt_path)


if __name__ == '__main__':
    tf.app.run()
