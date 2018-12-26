from kdgan import config
from kdgan import utils
from flags import flags
from dis_model import DIS
from std_model import STD
from data_utils import CIFAR

import tensorflow as tf
import math
import time

cifar = CIFAR(flags)
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))

tn_dis = DIS(flags, is_training=True)
tn_std = STD(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, is_training=False)
vd_std = STD(flags, is_training=False)

init_op = tf.global_variables_initializer()

# tot_params = 0
# for var in tf.trainable_variables():
#   num_params = 1
#   for dim in var.shape:
#     num_params *= dim.value
#   print('%-64s (%d params)' % (var.name, num_params))
#   tot_params += num_params
# print('%-64s (%d params)' % ('gan', tot_params))

def main(_):
  bst_acc = 0.0
  start_time = time.time()
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_dis.saver.restore(sess, flags.dis_model_ckpt)
    tn_std.saver.restore(sess, flags.std_model_ckpt)

    ini_dis = cifar.compute_acc(sess, vd_dis)
    ini_std = cifar.compute_acc(sess, vd_std)

    print('ini dis=%.4f ini std=%.4f' % (ini_dis, ini_std))

    batch_d, batch_s = -1, -1
    for epoch in range(flags.num_epoch):
      num_batch_d = math.ceil(flags.num_dis_epoch * flags.train_size / flags.batch_size)
      for _ in range(num_batch_d):
        batch_d += 1
        tn_image_d, label_dat_d = cifar.next_batch(sess)
        feed_dict = {tn_std.image_ph:tn_image_d}
        label_std_d = sess.run(tn_std.labels, feed_dict=feed_dict)
        sample_np_d, std_label_d = utils.gan_dis_sample(flags, label_dat_d, label_std_d)
        feed_dict = {
          tn_dis.image_ph:tn_image_d,
          tn_dis.std_sample_ph:sample_np_d,
          tn_dis.std_label_ph:std_label_d,
        }
        sess.run(tn_dis.gan_train, feed_dict=feed_dict)

        if (batch_d + 1) % eval_interval != 0:
          continue
        end_time = time.time()
        duration = (end_time - start_time) / 3600
        print('dis #batch=%d duration=%.4fh' % (batch_d, duration))
        # evaluate dis if necessary

      num_batch_s = math.ceil(flags.num_std_epoch * flags.train_size / flags.batch_size)
      for _ in range(num_batch_s):
        batch_s += 1
        tn_image_s, label_dat_s = cifar.next_batch(sess)
        feed_dict = {tn_std.image_ph:tn_image_s}
        label_gen_s = sess.run(tn_std.labels, feed_dict=feed_dict)
        sample_np_s = utils.generate_label(flags, label_dat_s, label_gen_s)
        feed_dict = {tn_dis.image_ph:tn_image_s, tn_dis.std_sample_ph:sample_np_s}
        reward_np_s = sess.run(tn_dis.std_rewards, feed_dict=feed_dict)
        feed_dict = {
          tn_std.image_ph:tn_image_s,
          tn_std.sample_ph:sample_np_s,
          tn_std.reward_ph:reward_np_s,
        }
        sess.run(tn_std.gan_train, feed_dict=feed_dict)

        if (batch_s + 1) % eval_interval != 0:
          continue
        acc = cifar.compute_acc(sess, vd_std)
        bst_acc = max(acc, bst_acc)
        end_time = time.time()
        duration = (end_time - start_time) / 3600
        print('gen #batch=%d acc=%.4f bst_acc=%.4f duration=%.4fh' % 
            (batch_s + 1, acc, bst_acc, duration))

        if acc < bst_acc:
          continue
        # save std if necessary
  tot_time = time.time() - start_time
  print('#cifar=%d final=%.4f et=%.0fs' % (flags.train_size, bst_acc, tot_time))

if __name__ == '__main__':
    tf.app.run()









