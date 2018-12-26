from kdgan import config

from datasets import dataset_utils
from os import path
from six.moves import urllib
import numpy as np
import tensorflow as tf
import gzip
import os
import sys

tf.app.flags.DEFINE_string('dataset_dir', None, '')
flags = tf.app.flags.FLAGS

# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABEL_FILENAME = 'train-labels-idx1-ubyte.gz'
_VALID_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_VALID_LABEL_FILENAME = 't10k-labels-idx1-ubyte.gz'

_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

# The names of the classes.
_CLASS_NAMES = [
  'zero',
  'one',
  'two',
  'three',
  'four',
  'five',
  'size',
  'seven',
  'eight',
  'nine',
]


def _extract_images(filename, num_images):
  print('Extracting images from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(
        _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  return data


def _extract_labels(filename, num_labels):
  print('Extracting labels from: ', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def _add_to_tfrecord(data_filepath, label_filepath, num_images, tfrecord_writer):
  images = _extract_images(data_filepath, num_images)
  labels = _extract_labels(label_filepath, num_images)

  label_cn = {}
  shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
  with tf.Graph().as_default():
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_png = tf.image.encode_png(image)

    with tf.Session('') as sess:
      for j in range(num_images):
        sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
        sys.stdout.flush()

        png_string = sess.run(encoded_png, feed_dict={image: images[j]})

        example = dataset_utils.image_to_tfexample(
            png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
        tfrecord_writer.write(example.SerializeToString())
        label_cn[labels[j]] = label_cn.get(labels[j], 0) + 1
  print('\nds=%s' % (path.basename(data_filepath)))
  for i in range(len(_CLASS_NAMES)):
    print('\t#%s=%d' % (_CLASS_NAMES[i], label_cn[i]))

def _get_output_filename(dataset_dir, split_name):
  filename = 'mnist_%s.tfrecord' % (split_name)
  return path.join(dataset_dir, filename)


def _download_dataset(dataset_dir):
  for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABEL_FILENAME,
                   _VALID_DATA_FILENAME,
                   _VALID_LABEL_FILENAME]:
    filepath = os.path.join(dataset_dir, filename)

    if not os.path.exists(filepath):
      print('Downloading file %s...' % filename)
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(_DATA_URL + filename,
                                               filepath,
                                               _progress)
      print()
      with tf.gfile.GFile(filepath) as f:
        size = f.size()
      print('Successfully downloaded', filename, size, 'bytes.')


def _clean_up_temporary_files(dataset_dir):
  for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABEL_FILENAME,
                   _VALID_DATA_FILENAME,
                   _VALID_LABEL_FILENAME]:
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

def main(_):
  if not flags.dataset_dir:
    raise ValueError('no --dataset_dir')
  dataset_dir = flags.dataset_dir
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  train_filepath = _get_output_filename(dataset_dir, 'train')
  valid_filepath = _get_output_filename(dataset_dir, 'valid')

  if tf.gfile.Exists(train_filepath) and tf.gfile.Exists(valid_filepath):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  _download_dataset(dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(train_filepath) as tfrecord_writer:
    data_filepath = os.path.join(dataset_dir, _TRAIN_DATA_FILENAME)
    label_filepath = os.path.join(dataset_dir, _TRAIN_LABEL_FILENAME)
    _add_to_tfrecord(data_filepath, label_filepath, 60000, tfrecord_writer)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(valid_filepath) as tfrecord_writer:
    data_filepath = os.path.join(dataset_dir, _VALID_DATA_FILENAME)
    label_filepath = os.path.join(dataset_dir, _VALID_LABEL_FILENAME)
    _add_to_tfrecord(data_filepath, label_filepath, 10000, tfrecord_writer)

  # Finally, write the labels file:
  label_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(label_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the MNIST dataset!')

if __name__ == '__main__':
  tf.app.run()




