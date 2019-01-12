import cifar10_utils

from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import math

class CIFAR_TF(object):
  def __init__(self, flags):
    cifar10_utils.maybe_download_and_extract()
    with tf.device('/cpu:0'):
      self.tn_image_ts, self.tn_label_ts = cifar10_utils.distorted_inputs()
      self.vd_image_ts, self.vd_label_ts = cifar10_utils.inputs(eval_data=True)
    self.vd_num_batch = int(math.ceil(flags.valid_size / flags.batch_size))

  def next_batch(self, sess):
    tn_image_np, tn_label_np = sess.run([self.tn_image_ts, self.tn_label_ts])
    return tn_image_np, tn_label_np

  def evaluate(self, sess, image_ph, hard_label_ph, accuracy, set_phase=False):
    acc_list = []
    for vd_batch in range(self.vd_num_batch):
      vd_image_np, vd_label_np = sess.run([self.vd_image_ts, self.vd_label_ts])
      if set_phase:
        feed_dict = {
          image_ph:vd_image_np,
          hard_label_ph:vd_label_np,
          K.learning_phase(): 0,
        }
      else:
        feed_dict = {image_ph:vd_image_np, hard_label_ph:vd_label_np}
      acc = sess.run(accuracy, feed_dict=feed_dict)
      acc_list.append(acc)
    acc = sum(acc_list) / len(acc_list)
    return acc

class KERAS_DG(object):
  def __init__(self, flags):
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_valid = x_valid.astype('float32') / 255.0
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_valid -= x_train_mean
    
    train_size, valid_size = x_train.shape[0], x_valid.shape[0]
    print('#train=%d #valid=%d' % (train_size, valid_size))

    datagen = ImageDataGenerator(
      # set input mean to 0 over the dataset
      featurewise_center=False,
      # set each sample mean to 0
      samplewise_center=False,
      # divide inputs by std of dataset
      featurewise_std_normalization=False,
      # divide each input by its std
      samplewise_std_normalization=False,
      # apply zca whitening
      zca_whitening=False,
      # randomly rotate images in the range (deg 0 to 180)
      rotation_range=0,
      # randomly shift images horizontally
      width_shift_range=0.1,
      # randomly shift images vertically
      height_shift_range=0.1,
      # randomly flip images
      horizontal_flip=True,
      # randomly flip images
      vertical_flip=False
    )
    datagen.fit(x_train)

    self.x_train, self.y_train = x_train, y_train
    self.x_valid, self.y_valid = x_valid, y_valid
    self.batch_size = flags.batch_size
    print('batch_size=%d' % (self.batch_size))
    self.datagen = datagen
    self.vd_num_batch = int(math.floor(flags.valid_size / flags.batch_size))

  def next_batch(self):
    tn_image_np, tn_label_np = next(self.datagen.flow(self.x_train, self.y_train, 
        batch_size=self.batch_size))
    return tn_image_np, tn_label_np

  def evaluate(self, sess, image_ph, hard_label_ph, accuracy):
    acc_list = []
    for vd_batch in range(self.vd_num_batch):
      start = vd_batch * self.batch_size
      vd_image_np = self.x_valid[start:start + self.batch_size]
      vd_label_np = self.y_valid[start:start + self.batch_size]
      feed_dict = {
        image_ph:vd_image_np,
        hard_label_ph:np.squeeze(vd_label_np),
        K.learning_phase(): 0,
      }
      acc = sess.run(accuracy, feed_dict=feed_dict)
      acc_list.append(acc)
    acc = sum(acc_list) / len(acc_list)
    return acc





