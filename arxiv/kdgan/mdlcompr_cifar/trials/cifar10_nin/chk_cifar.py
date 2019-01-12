from kdgan import config
import data_utils

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.initializers import RandomNormal  
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.layers import InputLayer, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2

batch_size = 128
weight_decay = 0.0001
dropout = 0.5

import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

image_ph = tf.placeholder(tf.float32, shape=(None, 32 * 32 * 3))
hard_label_ph = tf.placeholder(tf.float32, shape=(None, 10))

model = Sequential()
model.add(InputLayer(input_tensor=image_ph, input_shape=(None, 32 * 32 * 3)))
model.add(Reshape((32, 32, 3)))
model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

model.add(Dropout(dropout))

model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1),padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

model.add(Dropout(dropout))

model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
labels = model.output
print('label', type(labels), labels.shape)
accuracy = tf.reduce_mean(categorical_accuracy(hard_label_ph, labels))

reg_losses = model.losses
for reg_loss in reg_losses:
  print('reg_loss', type(reg_loss), reg_loss.shape)

hard_loss = tf.reduce_mean(categorical_crossentropy(labels, hard_label_ph))
print('hard_loss', type(hard_loss), hard_loss.shape)

pre_losses = [hard_loss]
pre_losses.extend(reg_losses)

pre_loss = tf.add_n(pre_losses)
# pre_update = tf.train.GradientDescentOptimizer(0.05).minimize(pre_loss)
pre_update = tf.train.GradientDescentOptimizer(0.1).minimize(hard_loss)

init_op = tf.global_variables_initializer()
sess.run(init_op)

from include.data import get_data_set
train_x, train_y, train_l = get_data_set()
test_x, test_y, test_l = get_data_set("test")
# print(type(test_x), type(test_y), type(test_l))

with sess.as_default():
  for tn_batch in range(10000):
    randidx = np.random.randint(len(train_x), size=batch_size)
    batch_xs = train_x[randidx]
    batch_ys = train_y[randidx]
    feed_dict = {
      image_ph:batch_xs,
      hard_label_ph:batch_ys,
      K.learning_phase(): 1,
    }
    # res = sess.run(pre_update, feed_dict=feed_dict)
    pre_update.run(feed_dict=feed_dict)

    if (tn_batch + 1) % 100 != 0:
      continue
    feed_dict = {
      image_ph:test_x[0:5000,],
      hard_label_ph:test_y[0:5000,],
      K.learning_phase(): 0,
    }
    acc = sess.run(accuracy, feed_dict=feed_dict)
    acc = accuracy.eval(feed_dict=feed_dict)
    print('#batch=%d acc=%.4f' % (tn_batch, acc))
