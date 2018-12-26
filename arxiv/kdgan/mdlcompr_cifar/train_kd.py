from flags import flags
from std_model import STD
from tch_model import TCH
from data_utils import CIFAR

import numpy as np
import tensorflow as tf
import math
import time

cifar = CIFAR(flags)

tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
print('#tn_batch=%d' % (tn_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))


tn_std = STD(flags, is_training=True)
tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_std = STD(flags, is_training=False)
vd_tch = TCH(flags, is_training=False)

init_op = tf.global_variables_initializer()

# tot_params = 0
# for var in tf.trainable_variables():
#   num_params = 1
#   for dim in var.shape:
#     num_params *= dim.value
#   print('%-64s (%d params)' % (var.name, num_params))
#   tot_params += num_params
# print('%-64s (%d params)' % (' '.join(['kd', flags.kd_model]), tot_params))

def main(_):
  bst_acc = 0.0
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_std.saver.restore(sess, flags.std_model_ckpt)
    ini_std = cifar.compute_acc(sess, vd_std)
    tf.logging.info('ini_std=%.4f' % (ini_std))
    tn_tch.saver.restore(sess, flags.tch_model_ckpt)
    ini_tch = cifar.compute_acc(sess, vd_tch)
    tf.logging.info('ini_tch=%.4f' % (ini_tch))

    start_time = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = cifar.next_batch(sess)

      # feed_dict = {tn_tch.image_ph:tn_image_np}
      # soft_logit_np = sess.run(tn_tch.logits, feed_dict=feed_dict)
      feed_dict = {vd_tch.image_ph:tn_image_np}
      soft_logit_np = sess.run(vd_tch.logits, feed_dict=feed_dict)

      # predictions = np.argmax(tn_label_np, axis=1)
      # groundtruth = np.argmax(soft_logit_np, axis=1)
      # accuracy = np.sum(predictions == groundtruth) / flags.batch_size
      # print('accuracy=%.4f' % (accuracy))

      feed_dict = {
        tn_std.image_ph:tn_image_np,
        tn_std.hard_label_ph:tn_label_np,
        tn_std.soft_logit_ph:soft_logit_np,
      }
      sess.run(tn_std.kd_train, feed_dict=feed_dict)

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
      # save model parameter if necessary
  tf.logging.info('#cifar=%d final=%.4f' % (flags.train_size, bst_acc))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

