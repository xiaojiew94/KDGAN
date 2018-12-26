from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../logs/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    print('images', type(images), images.shape)
    print('labels', type(labels), labels.shape)

    isum = tf.reduce_sum(images)
    lsum = tf.reduce_sum(labels)
    with tf.train.MonitoredTrainingSession() as mon_sess:
      start_time = time.time()
      for i in range(4000):
        res = mon_sess.run([isum, lsum])
      end_time = time.time()
      print('%.4fs' % (end_time - start_time))

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  train()


if __name__ == '__main__':
  tf.app.run()
