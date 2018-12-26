from kdgan import config
from flags import flags
import cifar10_utils

from datetime import datetime
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

def speed():
  with tf.device('/cpu:0'):
    image_ts, label_ts = cifar10_utils.distorted_inputs()
    isum_alp = tf.reduce_sum(image_ts)
    lsum_alp = tf.reduce_sum(label_ts)

    image_ph = tf.placeholder(tf.float32, shape=(None, 24, 24, 3))
    label_ph = tf.placeholder(tf.int32, shape=(None,))
    isum_bet = tf.reduce_sum(image_ph)
    lsum_bet = tf.reduce_sum(label_ph)

  with tf.train.MonitoredTrainingSession() as sess:
    start_time = time.time()
    for tn_batch in range(10000):
      isum_np, lsum_np = sess.run([isum_alp, lsum_alp])
      if (tn_batch + 1) % 2000 == 0:
        print('#alp=%d' % (tn_batch + 1))
    end_time = time.time()
    duration = end_time - start_time
    print('alp=%.4fs' % (duration))

    start_time = time.time()
    for tn_batch in range(10000):
      image_np, label_np = sess.run([image_ts, label_ts])
      feed_dict = {image_ph:image_np, label_ph:label_np}
      isum_np, lsum_np = sess.run([isum_bet, lsum_bet], feed_dict=feed_dict)
      if (tn_batch + 1) % 2000 == 0:
        print('#bet=%d' % (tn_batch + 1))
    end_time = time.time()
    duration = end_time - start_time
    print('bet=%.4fs' % (duration))

def train():
  eval_data = True
  with tf.device('/cpu:0'):
    image_cpu, label_cpu = cifar10_utils.inputs(eval_data=eval_data)
    isum_cpu, lsum_cpu = tf.reduce_sum(image_cpu), tf.reduce_sum(label_cpu)

  with tf.device('/gpu:0'):
    image_gpu, label_gpu = cifar10_utils.inputs(eval_data=eval_data)
    isum_gpu, lsum_gpu = tf.reduce_sum(image_gpu), tf.reduce_sum(label_gpu)

  with tf.train.MonitoredTrainingSession() as sess:
    start_time = time.time()
    for tn_batch in range(10000):
      isum_np, lsum_np = sess.run([isum_cpu, lsum_cpu])
      if (tn_batch + 1) % 2000 == 0:
        print('#cpu=%d' % (tn_batch + 1))
    end_time = time.time()
    duration = end_time - start_time
    print('cpu=%.4fs' % (duration))

    start_time = time.time()
    for tn_batch in range(10000):
      isum_np, lsum_np = sess.run([isum_gpu, lsum_gpu])
      if (tn_batch + 1) % 2000 == 0:
        print('#gpu=%d' % (tn_batch + 1))
    end_time = time.time()
    duration = end_time - start_time
    print('gpu=%.4fs' % (duration))


def main(argv=None):
  cifar10_utils.maybe_download_and_extract()
  train()

if __name__ == '__main__':
  tf.app.run()










