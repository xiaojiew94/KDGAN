from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
from tch_model import TCH
import data_utils

from os import path
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
import h5py
import time
import numpy as np
import tensorflow as tf

# dataset
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('channels', 1, '')
tf.app.flags.DEFINE_string('augmentation_type', 'affine', 'none|affine|align')
# model
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_string('checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('save_path', None, '')
tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
# optimization
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_epoch', 100, '')
tf.app.flags.DEFINE_string('optimizer', 'rmsprop', 'adam|sgd')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS


def main(_):
  # mnist = data_utils.read_data_sets(flags.dataset_dir, one_hot=True, validation_size=0)
  # print(mnist.train.images.shape, mnist.train.labels.shape)
  # train_images = np.tile(mnist.train.images, (flags.num_epoch, 1))
  # train_labels = np.tile(mnist.train.labels, (flags.num_epoch, 1))
  image_filename = 'images.npy'
  label_filename = 'labels.npy'
  mnist_filename = 'mnist.h5'
  # np.save(path.join(config.temp_dir, image_filename), train_images)
  # np.save(path.join(config.temp_dir, label_filename), train_labels)

  # start_time = time.time()
  # mnist = input_data.read_data_sets(flags.dataset_dir, one_hot=True, validation_size=0)
  # datagen = data_utils.Generator(flags.augmentation_type, mnist)
  # train_images = np.zeros((mnist.train.num_examples * flags.num_epoch, 28*28), dtype=np.float32)
  # train_labels = np.zeros((mnist.train.num_examples * flags.num_epoch, 10), dtype=np.float32)
  # start_index = 0
  # for tn_epoch in range(flags.num_epoch):
  #   print('tn_epoch=%d' % (tn_epoch))
  #   for tn_image_np, tn_label_np in datagen.generate(batch_size=flags.batch_size):
  #     end_index = start_index + tn_image_np.shape[0]
  #     train_images[start_index:end_index] = tn_image_np
  #     train_labels[start_index:end_index] = tn_label_np
  #     start_index = end_index
  # print(train_images.shape, train_labels.shape)
  # end_time = time.time()
  # print('time=%.2f' % (end_time - start_time))
  # start_time = time.time()
  # hf = h5py.File(path.join(config.temp_dir, mnist_filename), 'w')
  # hf.create_dataset('images', data=train_images)
  # hf.create_dataset('labels', data=train_labels)
  # hf.close()
  # end_time = time.time()
  # print('time=%.2f' % (end_time - start_time))

  # start_time = time.time()
  # train_images = np.load(path.join(config.temp_dir, image_filename))
  # train_labels = np.load(path.join(config.temp_dir, label_filename))
  # print(train_images.shape, train_labels.shape)
  # num_batch = int(train_images.shape[0] / flags.batch_size)
  # for batch in range(num_batch):
  #   start_index = batch * flags.batch_size
  #   end_index = (batch + 1) * flags.batch_size
  #   image_batch = train_images[start_index:end_index,:]
  #   label_batch = train_labels[start_index:end_index,:]
  # end_time = time.time()
  # print('time=%.2f' % (end_time - start_time))

  start_time = time.time()
  hf = h5py.File(path.join(config.temp_dir, mnist_filename), 'r')
  train_images = hf.get('images')
  train_labels = hf.get('labels')
  print(train_images.shape, train_labels.shape)
  num_batch = int(train_images.shape[0] / flags.batch_size)
  for batch in range(num_batch):
    start_index = batch * flags.batch_size
    end_index = (batch + 1) * flags.batch_size
    image_batch = train_images[start_index:end_index,:]
    label_batch = train_labels[start_index:end_index,:]
  end_time = time.time()
  print('time=%.2f' % (end_time - start_time))

  # start_time = time.time()
  # mnist = input_data.read_data_sets(flags.dataset_dir, one_hot=True, validation_size=0)
  # datagen = data_utils.Generator(flags.augmentation_type, mnist)
  # for tn_epoch in range(flags.num_epoch):
  #   for tn_image_np, tn_label_np in datagen.generate(batch_size=flags.batch_size):
  #     pass
  # end_time = time.time()
  # print('time=%.2f' % (end_time - start_time))

if __name__ == '__main__':
  tf.app.run()

# tn_dataset = data_utils.get_dataset(flags, is_training=True)
# vd_dataset = data_utils.get_dataset(flags, is_training=False)
# tn_gen = GEN(flags, tn_dataset, is_training=True)
# tn_tch = TCH(flags, tn_dataset, is_training=True)
# scope = tf.get_variable_scope()
# scope.reuse_variables()
# vd_gen = GEN(flags, vd_dataset, is_training=False)
# vd_tch = TCH(flags, vd_dataset, is_training=False)
# tn_image_bt, tn_label_bt = data_utils.generate_batch(flags, tn_dataset, is_training=True)
# vd_image_bt, vd_label_bt = data_utils.generate_batch(flags, vd_dataset, is_training=False)
# def count_param(scope_name):
#   tot_param = 0
#   for variable in tf.trainable_variables():
#     var_name = variable.name
#     if not var_name.startswith(scope_name):
#       continue
#     num_param = 1
#     for dim in variable.shape:
#       num_param *= dim.value
#     print('%-50s (%d params)' % (var_name, num_param))
#     tot_param += num_param
#   print('%s #param=%d' % (scope_name, tot_param))
# def main(_):
  # count_param(tn_gen.gen_scope)
  # count_param(tn_tch.tch_scope)
  # def parser(record):
  #   keys_to_features = {
  #     'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
  #     'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
  #     'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  #   }
  #   parsed = tf.parse_single_example(record, keys_to_features)
  #   return parsed['image/encoded'], parsed['image/class/label']
  # vd_label_cn = {}
  # valid_data_size = 10000
  # num_batch = int(valid_data_size / config.valid_batch_size)
  # valid_filepath = path.join(flags.dataset_dir, 'mnist_valid.tfrecord' )
  # dataset = tf.data.TFRecordDataset([valid_filepath])
  # dataset = dataset.map(parser)
  # dataset = dataset.batch(config.valid_batch_size)
  # iterator = dataset.make_one_shot_iterator()
  # image_bt, label_bt = iterator.get_next()
  # with tf.train.MonitoredTrainingSession() as sess:
  #   while not sess.should_stop():
  #     image_np, label_np = sess.run([image_bt, label_bt])
  #     for label in label_np:
  #       label = int(label)
  #       vd_label_cn[label] = vd_label_cn.get(label, 0) + 1
  #     print(image_np)
  #     exit()
  # for label in range(10):
  #   print('%d vd=%d' % (label, vd_label_cn.get(label, 0)))
  # valid_data_size = vd_dataset.num_samples
  # num_batch = int(valid_data_size / config.valid_batch_size)
  # with tf.train.MonitoredTrainingSession() as sess:
  #   tn_label_cn, vd_label_cn = {}, {}
  #   for i in range(num_batch):
  #     tn_image_np, tn_label_np = sess.run([tn_image_bt, tn_label_bt])
  #     vd_image_np, vd_label_np = sess.run([vd_image_bt, vd_label_bt])
      # for tn_label in np.argmax(tn_label_np, axis=1):
      #   tn_label_cn[tn_label] = tn_label_cn.get(tn_label, 0) + 1
      # for vd_label in np.argmax(vd_label_np, axis=1):
      #   vd_label_cn[vd_label] = vd_label_cn.get(vd_label, 0) + 1
      # for vd_image in vd_image_np:
      #   print(vd_image.shape)
      #   print(vd_image.flatten())
      #   break
      # exit()
  # for label in range(10):
  #   print('%d tn=%d vd=%d' % (label, tn_label_cn[label], vd_label_cn[label]))

