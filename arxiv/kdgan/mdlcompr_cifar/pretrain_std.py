from kdgan import config
from kdgan import utils
from flags import flags
from std_model import STD
from data_utils import CIFAR

from datetime import datetime
import numpy as np
import tensorflow as tf
import math
import time

cifar = CIFAR(flags)

tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
print('#tn_batch=%d' % (tn_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))

tn_std = STD(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_std = STD(flags, is_training=False)
init_op = tf.global_variables_initializer()

def main(_):
  bst_acc = 0.0
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start_time = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = cifar.next_batch(sess)
      feed_dict = {
        tn_std.image_ph:tn_image_np,
        tn_std.hard_label_ph:tn_label_np,
      }
      sess.run(tn_std.pre_train, feed_dict=feed_dict)
      if (tn_batch + 1) % eval_interval != 0 and (tn_batch + 1) != tn_num_batch:
        continue
      acc = cifar.compute_acc(sess, vd_std)
      bst_acc = max(acc, bst_acc)

      end_time = time.time()
      duration = end_time - start_time
      avg_time = duration / (tn_batch + 1)
      print('#batch=%d acc=%.4f time=%.4fs/batch est=%.4fh' % 
          (tn_batch + 1, bst_acc, avg_time, avg_time * tn_num_batch / 3600))

      if acc < bst_acc:
        continue
      tn_std.saver.save(utils.get_session(sess), flags.std_model_ckpt)
  print('#cifar=%d final=%.4f' % (flags.train_size, bst_acc))

if __name__ == '__main__':
  tf.app.run()










