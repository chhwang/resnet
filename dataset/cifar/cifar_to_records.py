from __future__ import print_function
import tensorflow as tf
from six.moves import urllib
import pickle
import numpy as np
import os
import sys
import tarfile

TRAIN_SIZE = 50000
TEST_SIZE = 10000
HEIGHT = 32
WIDTH = 32
DEPTH = 3

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def maybe_download_and_extract(data_dir, data_type):
    """Download and extract the tarball from Alex's website.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    url_prefix = 'https://www.cs.toronto.edu/~kriz/'
    url_postfix = [data_type + '-python.tar.gz']
    for postfix in url_postfix:
        data_url = url_prefix + postfix
        filename = data_url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(data_dir)

def read_files(data_dir, data_type, test):
    if data_type == 'cifar-10':
        if test:
            file_names = ['test_batch']
        else:
            file_names = ['data_batch_%d' % n for n in [1,2,3,4,5]]
        file_names = ['%s/cifar-10-batches-py/%s' % (data_dir,name)
                      for name in file_names]
        images_list = []
        labels_list = []
        for name in file_names:
            with open(name, 'rb') as f:
                data = pickle.load(f)
            images_list.append(data['data'])
            labels_list.append(np.asarray(data['labels']))
        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
    elif data_type == 'cifar-100':
        file_name = 'test' if test else 'train'
        file_name = '%s/cifar-100-python/%s' % (data_dir, file_name)
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        images = data['data']
        labels = np.asarray(data['fine_labels'])
    else:
        raise ValueError('Not supported data type: %s' % data_type)
    return images, labels

def main(unused_argv):
    data_dir = os.path.dirname(os.path.realpath(__file__))
    data_type = 'cifar-10'
    maybe_download_and_extract(data_dir, data_type)
    for test in [True, False]:
        num_examples = TEST_SIZE if test else TRAIN_SIZE
        images, labels = read_files(data_dir, data_type, test)
        istrain = 'test' if test else 'train'
        filename = '%s/%s-%s.tfrecords' % (data_dir,data_type,istrain)
        if os.path.exists(filename):
            print('File %s already exists.' % filename)
            continue
        print('Writing %s' % filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(HEIGHT),
                'width': _int64_feature(WIDTH),
                'depth': _int64_feature(DEPTH),
                'label': _int64_feature(labels[index]),
                'image': _bytes_feature(image)}))
            writer.write(example.SerializeToString())
        writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)
