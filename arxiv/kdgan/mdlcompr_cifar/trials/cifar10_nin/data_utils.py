from kdgan import config

from os import path
from six.moves import cPickle
from sys import stdout
from urllib import request
import numpy as np
import tensorflow as tf
import keras
import os
import pickle
import sys
import tarfile

# DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
NP_DTYPE = np.float32

class DataSet(object):
  def __init__(self, images, labels, dtype=NP_DTYPE, one_hot=True):
    assert images.shape[0] == labels.shape[0]
    if dtype == NP_DTYPE:
      images = images.astype(NP_DTYPE) / 255.0
    if one_hot:
      labels = dense_to_one_hot(labels, 10)
    # print('image', images.dtype, 'label', labels.dtype)
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes), dtype=NP_DTYPE)
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def maybe_download_and_extract():
  dest_directory = config.cifar_dir
  if not path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = path.join(dest_directory, filename)
  if not path.exists(filepath):
    def _progress(count, block_size, total_size):
      stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      stdout.flush()
    filepath, _ = request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = config.cifar_ext
  if not path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

# keras/datasets/cifar.py
def load_batch(fpath, label_key='labels'):
  with open(fpath, 'rb') as f:
    if sys.version_info < (3,):
      d = cPickle.load(f)
    else:
      d = cPickle.load(f, encoding='bytes')
      # decode utf8
      d_decoded = {}
      for k, v in d.items():
          d_decoded[k.decode('utf8')] = v
      d = d_decoded
  data = d['data']
  labels = d[label_key]
  # data = data.reshape(data.shape[0], 3, 32, 32)
  return data, labels

# keras/datasets/cifar10.py
def load_data():
  cifar_ext = config.cifar_ext

  num_train_samples = 50000
  train_images = np.empty((num_train_samples, 32 * 32 * 3), dtype='uint8')
  train_labels = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = path.join(cifar_ext, 'data_batch_' + str(i))
    (train_images[(i - 1) * 10000: i * 10000, :],
     train_labels[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

  fpath = path.join(cifar_ext, 'test_batch')
  valid_images, valid_labels = load_batch(fpath)

  train_labels = np.reshape(train_labels, (len(train_labels), 1))
  valid_labels = np.reshape(valid_labels, (len(valid_labels), 1))

  train = DataSet(train_images, train_labels)
  valid = DataSet(valid_images, valid_labels)
  return train, valid

class CIFAR(object):
  def __init__(self):
    maybe_download_and_extract()
    self.train, self.valid = load_data()
    # print('train image', self.train.images.shape, self.train.images.dtype)
    # print('train label', self.train.labels.shape, self.train.labels.dtype)