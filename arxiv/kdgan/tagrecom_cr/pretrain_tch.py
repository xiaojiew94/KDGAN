from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from tch_model import TCH
import data_utils

from os import path
from tensorflow.contrib import slim
import os
import pickle
import time
import numpy as np
import tensorflow as tf

tn_data_size = utils.get_tn_size(flags.dataset)
tn_num_batch = int(flags.num_epoch * tn_data_size / flags.batch_size)
eval_interval = int(tn_data_size / flags.batch_size)
print('#tn_size=%d #tn_batch=%d' % (tn_data_size, tn_num_batch))

tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_tch = TCH(flags, is_training=False)

tf.summary.scalar(tn_tch.learning_rate.name, tn_tch.learning_rate)
tf.summary.scalar(tn_tch.pre_loss.name, tn_tch.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

yfccdata = data_utils.YFCCDATA(flags)
yfcceval = data_utils.YFCCEVAL(flags)

def main(_):
  best_prec = 0.0
  prec_list = []
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start = time.time()
    for tn_batch in range(tn_num_batch):
      _, tn_text_np, tn_label_np = yfccdata.next_batch(flags, sess)
      feed_dict = {tn_tch.text_ph:tn_text_np, tn_tch.hard_label_ph:tn_label_np}
      _, summary = sess.run([tn_tch.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      prec = yfcceval.compute_prec(flags, sess, vd_tch)
      prec_list.append(prec)
      if (tn_batch + 1) % eval_interval != 0:
          continue
      best_prec = max(prec, best_prec)
      tot_time = time.time() - start
      global_step = sess.run(tn_tch.global_step)
      avg_time = (tot_time / global_step) * (tn_data_size / flags.batch_size)
      print('#%08d prec@%d=%.4f best=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (global_step, flags.cutoff, prec, best_prec, tot_time, avg_time))

      if prec < best_prec:
        continue
      tn_tch.saver.save(utils.get_session(sess), flags.tch_model_ckpt)
  tot_time = time.time() - start
  print('best@%d=%.4f et=%.0fs' % (flags.cutoff, best_prec, tot_time))

  utils.create_pardir(flags.all_learning_curve_p)
  pickle.dump(prec_list, open(flags.all_learning_curve_p, 'wb'))

if __name__ == '__main__':
  tf.app.run()