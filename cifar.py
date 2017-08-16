import tensorflow as tf
import os

TRAIN_SIZE = 50000
TEST_SIZE = 10000
HEIGHT = 32
WIDTH = 32
DEPTH = 3


def inputs(data_dir, batch_size, test=False):
    path = os.path.join(
        data_dir,
        'cifar-10-%s.tfrecords' % ('test' if test else 'train'))
    with tf.name_scope('inputs'):
        filename_queue = tf.train.string_input_producer([path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features =  tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([HEIGHT*WIDTH*DEPTH])
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [DEPTH,HEIGHT,WIDTH])
        image = tf.transpose(image, [1,2,0])
        label = tf.cast(features['label'], tf.int32)

        if not test:
            image = tf.pad(image, [[4,4,],[4,4,],[0,0,]], 'REFLECT')
            image = tf.random_crop(image, [HEIGHT,WIDTH,DEPTH])
            image = tf.image.random_flip_left_right(image)

        image = tf.image.per_image_standardization(image)

        # feed inputs in a shuffle queue
        if not test:
            image_batch, label_batch = tf.train.shuffle_batch(
                [image,label],
                batch_size=batch_size,
                capacity=TRAIN_SIZE,
                min_after_dequeue=int(TRAIN_SIZE*0.2),
                num_threads=1,
                allow_smaller_final_batch=False)
        else:
            image_batch, label_batch = tf.train.batch(
                [image,label],
                batch_size=batch_size,
                capacity=TEST_SIZE,
                num_threads=1,
                allow_smaller_final_batch=False)
        return image_batch, label_batch
