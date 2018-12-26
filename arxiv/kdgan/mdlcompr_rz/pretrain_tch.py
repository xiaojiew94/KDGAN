from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from tch_model import TCH
import data_utils

from os import path
from tensorflow.contrib import slim
import pickle
import time
import numpy as np
import tensorflow as tf

# tn_dataset = data_utils.get_dataset(flags, is_training=True)
# vd_dataset = data_utils.get_dataset(flags, is_training=False)
# tn_image_bt, tn_label_bt = data_utils.generate_batch(flags, tn_dataset, is_training=True)
# vd_image_bt, vd_label_bt = data_utils.generate_batch(flags, vd_dataset, is_training=False)

mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
tn_size, vd_size = mnist.train.num_examples, mnist.test.num_examples
print('tn size=%d vd size=%d' % (tn_size, vd_size))
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = max(int(tn_size / flags.batch_size), 1)
print('ev #interval=%d' % (eval_interval))

tn_tch = TCH(flags, mnist.train, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_tch = TCH(flags, mnist.test, is_training=False)

tot_params = 0
for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))
  tot_params += num_params
print('%-50s (%d params)' % (flags.tch_model_name, tot_params))
# input()

tf.summary.scalar(tn_tch.learning_rate.name, tn_tch.learning_rate)
tf.summary.scalar(tn_tch.pre_loss.name, tn_tch.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

def main(_):
  bst_acc = 0.0
  acc_list = []
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = mnist.train.next_batch(flags.batch_size)
      feed_dict = {
        tn_tch.image_ph:tn_image_np,
        tn_tch.hard_label_ph:tn_label_np
      }
      _, summary = sess.run([tn_tch.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if flags.collect_cr_data:
        feed_dict = {
          vd_tch.image_ph:mnist.test.images,
          vd_tch.hard_label_ph:mnist.test.labels,
        }
        acc = sess.run(vd_tch.accuracy, feed_dict=feed_dict)
        acc_list.append(acc)
        if (tn_batch + 1) % eval_interval != 0:
          continue
      else:
        if (tn_batch + 1) % eval_interval != 0:
          continue
        feed_dict = {
          vd_tch.image_ph:mnist.test.images,
          vd_tch.hard_label_ph:mnist.test.labels,
        }
        acc = sess.run(vd_tch.accuracy, feed_dict=feed_dict)

      bst_acc = max(acc, bst_acc)
      tot_time = time.time() - start
      global_step = sess.run(tn_tch.global_step)
      avg_time = (tot_time / global_step) * (mnist.train.num_examples / flags.batch_size)
      print('#%08d curacc=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (tn_batch, acc, bst_acc, tot_time, avg_time))

      if acc < bst_acc:
        continue
      tn_tch.saver.save(utils.get_session(sess), flags.tch_model_ckpt)
  tot_time = time.time() - start
  print('#mnist=%d bstacc=%.4f et=%.0fs' % (tn_size, bst_acc, tot_time))

  if flags.collect_cr_data:
    utils.create_pardir(flags.all_learning_curve_p)
    pickle.dump(acc_list, open(flags.all_learning_curve_p, 'wb'))

if __name__ == '__main__':
  tf.app.run()






