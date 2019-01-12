from kdgan import config, metric, utils
from gen_model import GEN
from tch_model import TCH

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_float('gen_weight_decay', 0.001, 'l2 coefficient')
tf.app.flags.DEFINE_float('init_learning_rate', 0.05, '')
tf.app.flags.DEFINE_float('beta', 0.3, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0, '')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_float('tch_weight_decay', 0.00001, 'l2 coefficient')

tf.app.flags.DEFINE_integer('cutoff', 3, '')
tf.app.flags.DEFINE_integer('embedding_size', 10, '')
tf.app.flags.DEFINE_integer('feature_size', 4096, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')

tf.app.flags.DEFINE_string('gen_model_ckpt', None, '')
tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('tch_model_ckpt', None, '')

flags = tf.app.flags.FLAGS

train_data_size = utils.get_tn_size(flags.dataset)
valid_data_size = utils.get_vd_size(flags.dataset)
num_batch_t = int(flags.num_epoch * train_data_size / config.train_batch_size)
num_batch_v = int(valid_data_size / config.valid_batch_size)
eval_interval = int(train_data_size / config.train_batch_size)
print('tn:\t#batch=%d\nvd:\t#batch=%d\neval:\t#interval=%d' % (
    num_batch_t, num_batch_v, eval_interval))

global_step = tf.train.create_global_step()
gen_t = GEN(flags, is_training=True)
tch_t = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
gen_v = GEN(flags, is_training=False)
tch_v = TCH(flags, is_training=False)

def eval(sess, bt_list_v):
  user_bt_v, image_bt_v, text_bt_v, label_bt_v, file_bt_v = bt_list_v
  image_hit_v, text_hit_v = [], []
  for batch_v in range(num_batch_v):
    image_np_v, text_np_v, label_np_v = sess.run([image_bt_v, text_bt_v, label_bt_v])
    feed_dict = {gen_v.image_ph:image_np_v, tch_v.text_ph:text_np_v}
    
    image_logit_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
    image_hit_bt = metric.compute_hit(image_logit_v, label_np_v, flags.cutoff)
    image_hit_v.append(image_hit_bt)

    text_logit_v, = sess.run([tch_v.logits], feed_dict=feed_dict)
    text_hit_bt = metric.compute_hit(text_logit_v, label_np_v, flags.cutoff)
    text_hit_v.append(text_hit_bt)
  image_hit_v = np.mean(image_hit_v)
  text_hit_v = np.mean(text_hit_v)
  # print('img:\thit=%.4f\ntxt:\thit=%.4f' % (image_hit_v, text_hit_v))
  return image_hit_v

def main(_):
  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('{}\t({} params)'.format(variable.name, num_params))

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
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    gen_t.saver.restore(sess, flags.gen_model_ckpt)
    tch_t.saver.restore(sess, flags.tch_model_ckpt)
    with slim.queues.QueueRunners(sess):
      image_hit_v = eval(sess, bt_list_v)
      print('init\thit={0:.4f}'.format(image_hit_v))
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
        _, summary = sess.run([gen_t.kd_train_op, gen_t.summary_op], feed_dict=feed_dict)
        writer.add_summary(summary, batch_t)

        if (batch_t + 1) % eval_interval != 0:
            continue

        tot_time = time.time() - start
        image_hit_v = eval(sess, bt_list_v)
        print('#{0}\thit={1:.4f} {2:.0f}s'.format(batch_t, image_hit_v, tot_time))

        if image_hit_v < best_hit_v:
          continue
        best_hit_v = image_hit_v
  print('best hit={0:.4f}'.format(best_hit_v))

if __name__ == '__main__':
    tf.app.run()


