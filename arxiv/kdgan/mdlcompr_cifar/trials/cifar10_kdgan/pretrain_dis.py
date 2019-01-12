from kdgan import config
from kdgan import utils
from flags import flags
from std_model import STD
from data_utils import CIFAR_TF
import cifar10_utils

from keras import backend as K
from keras.initializers import Constant, TruncatedNormal
from keras.layers import add
from keras.layers import Activation, Conv2D, Dense, Flatten, GlobalAveragePooling2D, \
    Input, InputLayer, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.models import Model, Sequential
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2
import numpy as np
import tensorflow as tf
import math
import time

cifar = CIFAR_TF(flags)
tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
print('#tn_batch=%d' % (tn_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))

image_shape = (flags.batch_size, flags.image_size, flags.image_size, flags.channels)
image_ph = tf.placeholder(tf.float32, shape=image_shape)
hard_label_ph = tf.placeholder(tf.int32, shape=(flags.batch_size))

'''
model = Sequential()
model.add(InputLayer(input_tensor=image_ph, input_shape=image_shape))
model.add(Conv2D(64, (5, 5), 
    padding='same', 
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.05),
    kernel_regularizer=None,
    bias_initializer=Constant(value=0)))
model.add(MaxPooling2D((3, 3),
    strides=(2, 2)))
model.add(Conv2D(64, (5, 5),
    padding='same',
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.05),
    kernel_regularizer=None,
    bias_initializer=Constant(value=0.1)))
model.add(MaxPooling2D((3, 3),
    strides=(2, 2)))
model.add(Flatten())
model.add(Dense(384,
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.04),
    kernel_regularizer=l2(flags.tch_weight_decay),
    bias_initializer=Constant(value=0.1)))
model.add(Dense(192,
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.04),
    kernel_regularizer=l2(flags.tch_weight_decay),
    bias_initializer=Constant(value=0.1)))
model.add(Dense(10,
    activation=None,
    kernel_initializer=TruncatedNormal(stddev=1/192.0),
    kernel_regularizer=None,
    bias_initializer=Constant(value=0.0)))
logits = model.output
'''

weight_decay = flags.tch_weight_decay
def residual_network(img_input,classes_num=10,stack_n=5):
  def residual_block(x,o_filters,increase=False):
    stride = (1,1)
    if increase:
      stride = (2,2)
    o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
    conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay))(o1)
    o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
    conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay))(o2)
    if increase:
      projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(weight_decay))(o1)
      block = add([conv_2, projection])
    else:
      block = add([conv_2, x])
    return block
  # build model ( total layers = stack_n * 3 * 2 + 2 )
  # stack_n = 5 by default, total layers = 32
  # input: 32x32x3 output: 32x32x16
  x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
             kernel_initializer="he_normal",
             kernel_regularizer=l2(weight_decay))(img_input)
  # input: 32x32x16 output: 32x32x16
  for _ in range(stack_n):
    x = residual_block(x,16,False)
  # input: 32x32x16 output: 16x16x32
  x = residual_block(x,32,True)
  for _ in range(1,stack_n):
    x = residual_block(x,32,False)
  # input: 16x16x32 output: 8x8x64
  x = residual_block(x,64,True)
  for _ in range(1,stack_n):
    x = residual_block(x,64,False)
  x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D()(x)
  # input: 64 output: 10
  # x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
  #           kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = Dense(classes_num,activation=None,kernel_initializer="he_normal",
            kernel_regularizer=l2(weight_decay))(x)
  return x

img_rows = img_cols = flags.image_size
img_channels = flags.channels
num_classes = flags.num_label
stack_n = 3
# img_input = Input(shape=(img_rows,img_cols,img_channels), 
#     batch_shape=(flags.batch_size,img_rows,img_cols,img_channels),
#     tensor=image_ph)
img_input = Input(shape=(img_rows,img_cols,img_channels))
output    = residual_network(img_input,num_classes,stack_n)
resnet = Model(img_input, output)
logits = output

hard_loss = cifar10_utils.loss(logits, hard_label_ph)
regularization_losses = resnet.losses
print('#regularization_losses=%d' % len(regularization_losses))
pre_losses = [hard_loss]
pre_losses.extend(regularization_losses)
pre_loss = tf.add_n(pre_losses)

top_k_op = tf.nn.in_top_k(logits, hard_label_ph, 1)
accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))

global_step = tf.Variable(0, trainable=False)
pre_train = cifar10_utils.get_train_op(flags, pre_loss, global_step)

init_op = tf.global_variables_initializer()

# for variable in tf.trainable_variables():
#   print('variable.name', variable.name)

def main(argv=None):
  bst_acc = 0.0
  with tf.Session() as sess:
    sess.run(init_op)
    start_time = time.time()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
    try:
      for tn_batch in range(tn_num_batch):
        tn_image_np, tn_label_np = cifar.next_batch(sess)
        # feed_dict = {
        #   image_ph:tn_image_np,
        #   hard_label_ph:tn_label_np,
        #   K.learning_phase(): 1,
        # }
        feed_dict = {
          img_input:tn_image_np,
          hard_label_ph:tn_label_np,
          K.learning_phase(): 1,
        }
        sess.run(pre_train, feed_dict=feed_dict)
        if (tn_batch + 1) % eval_interval != 0 and (tn_batch + 1) != tn_num_batch:
          continue
        # acc = cifar.evaluate(sess, image_ph, hard_label_ph, accuracy, set_phase=True)
        acc = cifar.evaluate(sess, img_input, hard_label_ph, accuracy, set_phase=True)
        bst_acc = max(acc, bst_acc)

        end_time = time.time()
        duration = end_time - start_time
        avg_time = duration / (tn_batch + 1)
        print('#batch=%d acc=%.4f time=%.4fs/batch est=%.4fh' % 
            (tn_batch + 1, bst_acc, avg_time, avg_time * tn_num_batch / 3600))

        if acc < bst_acc:
          continue
        # tn_std.saver.save(utils.get_session(sess), flags.std_model_ckpt)
    except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads) 
  print('final=%.4f' % (bst_acc))

if __name__ == '__main__':
  tf.app.run()










