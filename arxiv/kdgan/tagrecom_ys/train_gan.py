from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS
from gen_model import GEN

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
tf.app.flags.DEFINE_string('gan_figure_data', None, '')
flags = tf.app.flags.FLAGS

train_data_size = utils.get_train_data_size(flags.dataset)
eval_interval = int(train_data_size / flags.batch_size)
print('eval:\t#interval=%d' % (eval_interval))

dis_t = DIS(flags, is_training=True)
gen_t = GEN(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
dis_v = DIS(flags, is_training=False)
gen_v = GEN(flags, is_training=False)

def main(_):
  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('%-50s (%d params)' % (variable.name, num_params))

  dis_summary_op = tf.summary.merge([
    tf.summary.scalar(dis_t.learning_rate.name, dis_t.learning_rate),
    tf.summary.scalar(dis_t.gan_loss.name, dis_t.gan_loss),
  ])
  gen_summary_op = tf.summary.merge([
    tf.summary.scalar(gen_t.learning_rate.name, gen_t.learning_rate),
    tf.summary.scalar(gen_t.gan_loss.name, gen_t.gan_loss),
  ])
  print(type(dis_summary_op), type(gen_summary_op))
  init_op = tf.global_variables_initializer()

  data_sources_t = utils.get_data_sources(flags, is_training=True)
  data_sources_v = utils.get_data_sources(flags, is_training=False)
  print('tn: #tfrecord=%d\nvd: #tfrecord=%d' % (len(data_sources_t), len(data_sources_v)))
  
  ts_list_d = utils.decode_tfrecord(flags, data_sources_t, shuffle=True)
  bt_list_d = utils.generate_batch(ts_list_d, flags.batch_size)
  user_bt_d, image_bt_d, text_bt_d, label_bt_d, file_bt_d = bt_list_d

  ts_list_g = utils.decode_tfrecord(flags, data_sources_t, shuffle=True)
  bt_list_g = utils.generate_batch(ts_list_g, flags.batch_size)
  user_bt_g, image_bt_g, text_bt_g, label_bt_g, file_bt_g = bt_list_g

  ts_list_v = utils.decode_tfrecord(flags, data_sources_v, shuffle=False)
  bt_list_v = utils.generate_batch(ts_list_v, config.valid_batch_size)

  figure_data = []
  best_hit_v = -np.inf
  start = time.time()
  with tf.Session() as sess:
    sess.run(init_op)
    dis_t.saver.restore(sess, flags.dis_model_ckpt)
    gen_t.saver.restore(sess, flags.gen_model_ckpt)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    with slim.queues.QueueRunners(sess):
      hit_v = utils.evaluate_image(flags, sess, gen_v, bt_list_v)
      print('init hit=%.4f' % (hit_v))

      batch_d, batch_g = -1, -1
      for epoch in range(flags.num_epoch):
        for dis_epoch in range(flags.num_dis_epoch):
          print('epoch %03d dis_epoch %03d' % (epoch, dis_epoch))
          num_batch_d = math.ceil(train_data_size / flags.batch_size)
          for _ in range(num_batch_d):
            batch_d += 1
            image_np_d, label_dat_d = sess.run([image_bt_d, label_bt_d])
            feed_dict = {gen_t.image_ph:image_np_d}
            label_gen_d, = sess.run([gen_t.labels], feed_dict=feed_dict)
            sample_np_d, label_np_d = utils.gan_dis_sample(
                flags, label_dat_d, label_gen_d)
            feed_dict = {
              dis_t.image_ph:image_np_d,
              dis_t.sample_ph:sample_np_d,
              dis_t.dis_label_ph:label_np_d,
            }
            _, summary_d = sess.run([dis_t.gan_update, dis_summary_op], 
                feed_dict=feed_dict)
            writer.add_summary(summary_d, batch_d)

        for gen_epoch in range(flags.num_gen_epoch):
          print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
          num_batch_g = math.ceil(train_data_size / flags.batch_size)
          for _ in range(num_batch_g):
            batch_g += 1
            image_np_g, label_dat_g = sess.run([image_bt_g, label_bt_g])
            feed_dict = {gen_t.image_ph:image_np_g}
            label_gen_g, = sess.run([gen_t.labels], feed_dict=feed_dict)
            sample_np_g = utils.generate_label(
                flags, label_dat_g, label_gen_g)
            feed_dict = {
              dis_t.image_ph:image_np_g,
              dis_t.sample_ph:sample_np_g,
            }
            reward_np_g, = sess.run([dis_t.rewards], feed_dict=feed_dict)
            feed_dict = {
              gen_t.image_ph:image_np_g,
              gen_t.sample_ph:sample_np_g,
              gen_t.reward_ph:reward_np_g,
            }
            _, summary_g = sess.run([gen_t.gan_update, gen_summary_op], 
                feed_dict=feed_dict)
            writer.add_summary(summary_g, batch_g)
            
            # if (batch_g + 1) % eval_interval != 0:
            #   continue
            # hit_v = utils.evaluate(flags, sess, gen_v, bt_list_v)
            # tot_time = time.time() - start
            # print('#%08d hit=%.4f %06ds' % (batch_g, hit_v, int(tot_time)))
            # if hit_v <= best_hit_v:
            #   continue
            # best_hit_v = hit_v
            # print('best hit=%.4f' % (best_hit_v))
        hit_v = utils.evaluate_image(flags, sess, gen_v, bt_list_v)
        tot_time = time.time() - start
        print('#%03d curbst=%.4f %.0fs' % (epoch, hit_v, tot_time))
        figure_data.append((epoch, hit_v))
        if hit_v <= best_hit_v:
          continue
        best_hit_v = hit_v
  print('bsthit=%.4f' % (best_hit_v))

  utils.create_if_nonexist(os.path.dirname(flags.gan_figure_data))
  fout = open(flags.gan_figure_data, 'w')
  for epoch, hit_v in figure_data:
    fout.write('%d\t%.4f\n' % (epoch, hit_v))
  fout.close()

if __name__ == '__main__':
  tf.app.run()






