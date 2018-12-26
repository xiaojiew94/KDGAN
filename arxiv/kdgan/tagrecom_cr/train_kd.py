from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
from tch_model import TCH

import math
import os
import time
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('dataset', None, '')
tf.app.flags.DEFINE_integer('num_label', 100, '')
# evaluation
tf.app.flags.DEFINE_integer('cutoff', 3, '')
# image model
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_integer('feature_size', 4096, '')
tf.app.flags.DEFINE_string('model_name', None, '')
# training
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_epoch', 20, '')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, '')
tf.app.flags.DEFINE_float('end_learning_rate', 0.00001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 20.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
# dis model
tf.app.flags.DEFINE_float('dis_weight_decay', 0.0, 'l2 coefficient')
tf.app.flags.DEFINE_string('dis_model_ckpt', None, '')
tf.app.flags.DEFINE_integer('num_dis_epoch', 10, '')
# gen model
tf.app.flags.DEFINE_float('kd_lamda', 0.3, '')
tf.app.flags.DEFINE_float('gen_weight_decay', 0.001, 'l2 coefficient')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_string('gen_model_ckpt', None, '')
tf.app.flags.DEFINE_integer('num_gen_epoch', 5, '')
# tch model
tf.app.flags.DEFINE_float('tch_weight_decay', 0.00001, 'l2 coefficient')
tf.app.flags.DEFINE_integer('embedding_size', 10, '')
tf.app.flags.DEFINE_string('tch_model_ckpt', None, '')
tf.app.flags.DEFINE_integer('num_tch_epoch', 5, '')
# kdgan
tf.app.flags.DEFINE_integer('num_negative', 1, '')
tf.app.flags.DEFINE_integer('num_positive', 1, '')
flags = tf.app.flags.FLAGS

train_data_size = utils.get_train_data_size(flags.dataset)
num_batch_t = int(flags.num_epoch * train_data_size / config.train_batch_size)
eval_interval = int(train_data_size / config.train_batch_size)
print('tn:\t#batch=%d\neval:\t#interval=%d' % (num_batch_t, eval_interval))

gen_t = GEN(flags, is_training=True)
tch_t = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
gen_v = GEN(flags, is_training=False)
tch_v = TCH(flags, is_training=False)

def main(_):
  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('%-50s (%d params)' % (variable.name, num_params))

  tf.summary.scalar(gen_t.learning_rate.name, gen_t.learning_rate)
  tf.summary.scalar(gen_t.kd_loss.name, gen_t.kd_loss)
  summary_op = tf.summary.merge_all()
  init_op = tf.global_variables_initializer()

  data_sources_t = utils.get_data_sources(flags, is_training=True)
  data_sources_v = utils.get_data_sources(flags, is_training=False)
  print('tn: #tfrecord=%d\nvd: #tfrecord=%d' % (len(data_sources_t), len(data_sources_v)))
  
  ts_list_t = utils.decode_tfrecord(flags, data_sources_t, shuffle=True)
  ts_list_v = utils.decode_tfrecord(flags, data_sources_v, shuffle=False)
  bt_list_t = utils.generate_batch(ts_list_t, config.train_batch_size)
  bt_list_v = utils.generate_batch(ts_list_v, config.valid_batch_size)
  user_bt_t, image_bt_t, text_bt_t, label_bt_t, file_bt_t = bt_list_t

  best_hit_v = -np.inf
  start = time.time()
  with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    gen_t.saver.restore(sess, flags.gen_model_ckpt)
    tch_t.saver.restore(sess, flags.tch_model_ckpt)
    with slim.queues.QueueRunners(sess):
      hit_v = utils.evaluate(flags, sess, gen_v, bt_list_v)
      print('init hit=%.4f' % (hit_v))

      for batch_t in range(num_batch_t):
        image_np_t, text_np_t, hard_labels = sess.run([image_bt_t, text_bt_t, label_bt_t])
        # print('hard labels:\t{}'.format(hard_labels.shape))
        # print(np.argsort(-hard_labels[0,:])[:10])

        feed_dict = {tch_t.text_ph:text_np_t}
        soft_labels, = sess.run([tch_t.labels], feed_dict=feed_dict)
        # print('soft labels:\t{}'.format(soft_labels.shape))
        # print(np.argsort(-soft_labels[0,:])[:10])

        feed_dict = {
          gen_t.image_ph:image_np_t,
          gen_t.hard_label_ph:hard_labels,
          gen_t.soft_label_ph:soft_labels,
        }
        _, summary = sess.run([gen_t.kd_update, summary_op], feed_dict=feed_dict)
        writer.add_summary(summary, batch_t)

        if (batch_t + 1) % eval_interval != 0:
            continue
        hit_v = utils.evaluate(flags, sess, gen_v, bt_list_v)
        tot_time = time.time() - start
        print('#%08d hit=%.4f %06ds' % (batch_t, hit_v, int(tot_time)))
        if hit_v <= best_hit_v:
          continue
        best_hit_v = hit_v
        print('best hit=%.4f' % (best_hit_v))
  print('best hit=%.4f' % (best_hit_v))

if __name__ == '__main__':
    tf.app.run()


