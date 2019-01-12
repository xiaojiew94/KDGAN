from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from dis_model import DIS
import data_utils

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tn_size = utils.get_tn_size(flags.dataset)
vd_size = utils.get_vd_size(flags.dataset)
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
vd_num_batch = int(vd_size / config.valid_batch_size)
eval_interval = int(tn_size / flags.batch_size)
print('#tn_size=%d #vd_size=%d' % (tn_size, vd_size))
eval_interval = int(tn_size / flags.batch_size)

tn_dis = DIS(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, is_training=False)

tf.summary.scalar(tn_dis.learning_rate.name, tn_dis.learning_rate)
tf.summary.scalar(tn_dis.pre_loss.name, tn_dis.pre_loss)
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
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, _, tn_label_np = yfccdata.next_batch(flags, sess)
      feed_dict = {tn_dis.image_ph:tn_image_np, tn_dis.hard_label_ph:tn_label_np}
      _, summary = sess.run([tn_dis.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if (tn_batch + 1) % eval_interval != 0:
          continue
      prec = yfcceval.compute_prec(flags, sess, vd_dis)
      best_prec = max(prec, best_prec)
      tot_time = time.time() - start
      global_step = sess.run(tn_dis.global_step)
      avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
      print('#%08d prec@%d=%.4f best=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (global_step, flags.cutoff, prec, best_prec, tot_time, avg_time))

      if prec < best_prec:
        continue
      tn_dis.saver.save(utils.get_session(sess), flags.dis_model_ckpt)
  print('best hit=%.4f' % (best_prec))

if __name__ == '__main__':
  tf.app.run()