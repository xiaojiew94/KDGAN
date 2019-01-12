from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from gen_model import GEN
import data_utils

import pickle
import time
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import slim

mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
tn_size, vd_size = mnist.train.num_examples, mnist.test.num_examples
print('tn size=%d vd size=%d' % (tn_size, vd_size))
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
vd_num_batch = int(tn_size / config.valid_batch_size)
print('tn #batch=%d vd #batch=%d' % (tn_num_batch, vd_num_batch))
eval_interval = max(int(tn_size / flags.batch_size), 1)
print('ev #interval=%d' % (eval_interval))

tn_gen = GEN(flags, mnist.train, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, mnist.test, is_training=False)

tot_params = 0
for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))
  tot_params += num_params
print('%-50s (%d params)' % ('mlp', tot_params))
exit()

if flags.log_model_size:
  with open(flags.size_file, 'a') as fout:
    k_params = tot_params / 1024
    m_params = k_params / 1024
    fout.write('gen %d %d %.2fK %.2fM\n' % (
        flags.hidden_size, tot_params, k_params, m_params))
  exit()

tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.pre_loss.name, tn_gen.pre_loss)
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
      feed_dict = {tn_gen.image_ph:tn_image_np, tn_gen.hard_label_ph:tn_label_np}
      _, summary = sess.run([tn_gen.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if flags.log_accuracy:
        feed_dict = {
          vd_gen.image_ph:mnist.test.images,
          vd_gen.hard_label_ph:mnist.test.labels,
        }
        acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)
        acc_list.append(acc)
        if (tn_batch + 1) % eval_interval != 0:
          continue
      else:
        if (tn_batch + 1) % eval_interval != 0:
          continue
        feed_dict = {
          vd_gen.image_ph:mnist.test.images,
          vd_gen.hard_label_ph:mnist.test.labels,
        }
        acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)

      bst_acc = max(acc, bst_acc)
      tot_time = time.time() - start
      global_step = sess.run(tn_gen.global_step)
      avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
      print('#%08d curacc=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (tn_batch, acc, bst_acc, tot_time, avg_time))

      if acc < bst_acc:
        continue
      tn_gen.saver.save(utils.get_session(sess), flags.gen_model_ckpt)
  tot_time = time.time() - start
  print('#mnist=%d bstacc=%.4f et=%.0fs' % (tn_size, bst_acc, tot_time))

  if flags.log_accuracy:
    utils.create_pardir(flags.all_learning_curve_p)
    pickle.dump(acc_list, open(flags.all_learning_curve_p, 'wb'))

if __name__ == '__main__':
  tf.app.run()
