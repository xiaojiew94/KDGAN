from kdgan import config
from kdgan import utils


from datasets import dataset_factory
from nets import nets_factory
from os import path
from preprocessing import preprocessing_factory
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib import slim
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
import gzip
import hashlib
import numpy as np
import tensorflow as tf



label_size = 23
legend_size = 17
tick_size = 19
line_width = 1.5
marker_size = 16
broken_length = 0.015
length_3rd = 6.66
length_2nd = length_3rd * 0.49 / 0.33
# fig_height = 4.80
conv_height = 3.60
tune_height = 3.20
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


class AffineGenerator():
  def __init__(self, mnist):
    from keras.preprocessing.image import ImageDataGenerator
    
    self.mnist = mnist
    self.datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
    self.train_x = np.reshape(self.mnist.train.images, [-1, 28, 28, 1])
    self.train_y = self.mnist.train.labels

  def generate(self, batch_size=64):
    cnt = 0
    batch_n = self.train_x.shape[0] // batch_size
    for x, y in self.datagen.flow(self.train_x, self.train_y, batch_size=batch_size):
      ret_x = x.reshape(-1, 784)
      yield ret_x, y

      cnt += 1
      if cnt == batch_n:
        break


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  # print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  # print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


def preprocessing(images):
  images = images.astype(np.float32)
  images = np.multiply(images, 1.0 / 255.0)
  ## lenet preprocessing
  # images = np.subtract(images, 128.0)
  # images = np.multiply(images, 1.0 / 128.0)
  return images


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = preprocessing(images)
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

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
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


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   train_size=50000,
                   valid_size=10000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  if not source_url:  # empty string check
    source_url = DEFAULT_SOURCE_URL

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   source_url + TRAIN_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    train_images = extract_images(f)

  # train_num_examples = train_images.shape[0]
  # ikeys = set()
  # for i in range(train_num_examples):
  #   inonzero_m, inonzero_n, inonzero_l = train_images[i].nonzero()
  #   ikey = []
  #   for m, n, l in zip(inonzero_m, inonzero_n, inonzero_l):
  #     ikey.append(str(train_images[i, m, n, l]))
  #   ikey = '_'.join(ikey)
  #   ikey = hashlib.sha224(ikey)
  #   ikey = ikey.hexdigest()
  #   # print('%d %s' % (i, ikey))
  #   ikeys.add(ikey)
  # print('#ikey=%d' % (len(ikeys)))

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   source_url + TRAIN_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   source_url + TEST_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   source_url + TEST_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= train_size <= len(train_images):
    raise ValueError(
        'train size should be between 0 and {}. Received: {}.'
        .format(len(train_images), train_size))

  if not 0 <= valid_size <= len(train_images):
    raise ValueError(
        'valid size should be between 0 and {}. Received: {}.'
        .format(len(train_images), valid_size))

  valid_images = train_images[:valid_size]
  valid_labels = train_labels[:valid_size]
  # train_images = train_images[valid_size:]
  # train_labels = train_labels[valid_size:]
  train_images = train_images[len(train_images) - train_size:]
  train_labels = train_labels[len(train_labels) - train_size:]
  # print('train image={} label={}'.format(train_images.shape, train_labels.shape))
  # train_label_cn = {}
  # for train_label in train_labels:
  #   train_label = train_label.nonzero()[0][0]
  #   train_label_cn[train_label] = train_label_cn.get(train_label, 0) + 1
  # for train_label, count in train_label_cn.items():
  #   print('train label=%d count=%d' % (train_label, count))

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(valid_images, valid_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)


def get_dataset(flags, is_training=True):
  if is_training:
    split_name = 'train'
  else:
    split_name = 'valid'
  dataset_name = path.basename(flags.dataset_dir)
  dataset = dataset_factory.get_dataset(dataset_name, split_name, flags.dataset_dir)
  print('%s #dataset=%d' % (split_name, dataset.num_samples))
  return dataset


def generate_batch(flags, dataset, is_training=True):
  if is_training:
    batch_size = flags.batch_size
    shuffle = True
    ## cause not to traverse valid data
    num_readers = config.num_readers
  else:
    batch_size = config.valid_batch_size
    shuffle = False
    ## cause not to traverse valid data
    num_readers = 1
  print('#dataset=%d #batch=%03d shuffle=%s' % (dataset.num_samples, batch_size, shuffle))

  preprocessing = preprocessing_factory.get_preprocessing(flags.preprocessing_name,
      is_training=is_training)

  provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
      num_readers=num_readers,
      common_queue_capacity=20 * batch_size,
      common_queue_min=10 * batch_size,
      shuffle=shuffle)

  [image_ts, label_ts] = provider.get(['image', 'label'])
  # image_ts = preprocessing(image_ts, flags.image_size, flags.image_size)
  image_ts = tf.to_float(image_ts)
  image_ts = tf.div(image_ts, 255.0)
  image_bt, label_bt = tf.train.batch(
      [image_ts, label_ts],
      num_threads=config.num_preprocessing_threads,
      capacity=5 * batch_size,
      batch_size=batch_size,)
  # label_bt = slim.one_hot_encoding(label_bt, dataset.num_classes)
  return image_bt, label_bt



