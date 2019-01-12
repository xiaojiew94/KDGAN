from kdgan import config
from kdgan import utils
from flags import flags
from data_utils import CIFAR_TF, KERAS_DG
import cifar10_utils

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import keras
import math
import os
import time

input_shape = (flags.image_size, flags.image_size, flags.channels)
n = 3
depth = n * 6 + 2
image_ph = Input(shape=input_shape)
hard_label_ph = tf.placeholder(tf.int32, shape=(flags.batch_size))

def resnet_layer(inputs,
      num_filters=16,
      kernel_size=3,
      strides=1,
      activation='relu',
      batch_normalization=True,
      conv_first=True):
  conv = Conv2D(num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=l2(1e-4))

  x = inputs
  if conv_first:
    x = conv(x)
    if batch_normalization:
      x = BatchNormalization()(x)
    if activation is not None:
      x = Activation(activation)(x)
  else:
    if batch_normalization:
      x = BatchNormalization()(x)
    if activation is not None:
      x = Activation(activation)(x)
    x = conv(x)
  return x


def resnet_v1(input_shape, depth, num_classes=10):
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  # Start model definition.
  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  x = resnet_layer(inputs=image_ph)
  # Instantiate the stack of residual units
  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(inputs=x,
          num_filters=num_filters,
          strides=strides)
      y = resnet_layer(inputs=y,
          num_filters=num_filters,
          activation=None)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match
        # changed dims
        x = resnet_layer(inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            activation=None,
            batch_normalization=False)
      x = keras.layers.add([x, y])
      x = Activation('relu')(x)
    num_filters *= 2

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = AveragePooling2D(pool_size=8)(x)
  y = Flatten()(x)
  # outputs = Dense(num_classes,
  #     activation='softmax',
  #     kernel_initializer='he_normal')(y)
  outputs = Dense(num_classes,
      activation=None,
      kernel_initializer='he_normal')(y)

  # Instantiate model.
  model = Model(inputs=image_ph, outputs=outputs)
  return model

model = resnet_v1(input_shape=input_shape, depth=depth)
logits = model.output
# print('logits', logits.shape, logits.dtype)

regularization_losses = model.losses
# print('#regularization_losses=%d' % len(regularization_losses))
hard_loss = cifar10_utils.loss(logits, hard_label_ph)
pre_losses = [hard_loss]
pre_losses.extend(regularization_losses)
pre_loss = tf.add_n(pre_losses)

top_k_op = tf.nn.in_top_k(logits, hard_label_ph, 1)
accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))

global_step = tf.Variable(0, trainable=False)
pre_train = cifar10_utils.get_train_op(flags, pre_loss, global_step)

init_op = tf.global_variables_initializer()

keras_dg = KERAS_DG(flags)

tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
print('#tn_batch=%d' % (tn_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

def main(argv=None):
  bst_acc = 0.0
  with tf.Session() as sess:
    sess.run(init_op)
    start_time = time.time()

    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = keras_dg.next_batch()
      feed_dict = {
        image_ph:tn_image_np,
        hard_label_ph:np.squeeze(tn_label_np),
        K.learning_phase(): 1,
      }
      sess.run(pre_train, feed_dict=feed_dict)

      if (tn_batch + 1) % eval_interval != 0 and (tn_batch + 1) != tn_num_batch:
        continue
      # acc = keras_dg.evaluate(sess, image_ph, hard_label_ph, accuracy)
      _, acc = model.evaluate(keras_dg.x_valid, keras_dg.y_valid)
      bst_acc = max(acc, bst_acc)

      end_time = time.time()
      duration = end_time - start_time
      avg_time = duration / (tn_batch + 1)
      print('#batch=%d acc=%.4f time=%.4fs/batch est=%.4fh' % 
          (tn_batch + 1, bst_acc, avg_time, avg_time * tn_num_batch / 3600))

      if acc < bst_acc:
        continue
  print('final=%.4f' % (bst_acc))




if __name__ == '__main__':
  tf.app.run()












