from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS
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
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0, '')
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
tf.app.flags.DEFINE_string('kdgan_figure_data', None, '')
tf.app.flags.DEFINE_string('kdgan_model_ckpt', None, '')
flags = tf.app.flags.FLAGS

train_data_size = utils.get_train_data_size(flags.dataset)
eval_interval = int(train_data_size / flags.batch_size)
print('eval:\t#interval=%d' % (eval_interval))
num_batch_per_epoch = math.ceil(train_data_size / flags.batch_size)

dis_t = DIS(flags, is_training=True)
gen_t = GEN(flags, is_training=True)
tch_t = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
dis_v = DIS(flags, is_training=False)
gen_v = GEN(flags, is_training=False)
tch_v = TCH(flags, is_training=False)

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
    tf.summary.scalar(gen_t.kdgan_loss.name, gen_t.kdgan_loss),
  ])
  tch_summary_op = tf.summary.merge([
    tf.summary.scalar(tch_t.learning_rate.name, tch_t.learning_rate),
    tf.summary.scalar(tch_t.kdgan_loss.name, tch_t.kdgan_loss),
  ])
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

  ts_list_t = utils.decode_tfrecord(flags, data_sources_t, shuffle=True)
  bt_list_t = utils.generate_batch(ts_list_t, flags.batch_size)
  user_bt_t, image_bt_t, text_bt_t, label_bt_t, file_bt_t = bt_list_t

  ts_list_v = utils.decode_tfrecord(flags, data_sources_v, shuffle=False)
  bt_list_v = utils.generate_batch(ts_list_v, config.valid_batch_size)
  
  figure_data = []
  best_hit_v = -np.inf
  start = time.time()
  with tf.Session() as sess:
    sess.run(init_op)
    dis_t.saver.restore(sess, flags.dis_model_ckpt)
    gen_t.saver.restore(sess, flags.gen_model_ckpt)
    tch_t.saver.restore(sess, flags.tch_model_ckpt)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    with slim.queues.QueueRunners(sess):
      gen_hit = utils.evaluate_image(flags, sess, gen_v, bt_list_v)
      tch_hit = utils.evaluate_text(flags, sess, tch_v, bt_list_v)
      print('hit gen=%.4f tch=%.4f' % (gen_hit, tch_hit))

      batch_d, batch_g, batch_t = -1, -1, -1
      for epoch in range(flags.num_epoch):
        for dis_epoch in range(flags.num_dis_epoch):
          print('epoch %03d dis_epoch %03d' % (epoch, dis_epoch))
          for _ in range(num_batch_per_epoch):
            #continue
            batch_d += 1
            image_d, text_d, label_dat_d = sess.run([image_bt_d, text_bt_d, label_bt_d])
            
            feed_dict = {gen_t.image_ph:image_d}
            label_gen_d, = sess.run([gen_t.labels], feed_dict=feed_dict)
            # print('gen label', label_gen_d.shape)
            feed_dict = {tch_t.text_ph:text_d, tch_t.image_ph:image_d}
            label_tch_d, = sess.run([tch_t.labels], feed_dict=feed_dict)
            # print('tch label', label_tch_d.shape)

            sample_d, label_d = utils.kdgan_dis_sample(flags, 
                label_dat_d, label_gen_d, label_tch_d)
            # print(sample_d.shape, label_d.shape)

            feed_dict = {
              dis_t.image_ph:image_d,
              dis_t.sample_ph:sample_d,
              dis_t.dis_label_ph:label_d,
            }
            _, summary_d = sess.run([dis_t.gan_update, dis_summary_op], 
                feed_dict=feed_dict)
            writer.add_summary(summary_d, batch_d)

        for tch_epoch in range(flags.num_tch_epoch):
          print('epoch %03d tch_epoch %03d' % (epoch, tch_epoch))
          for _ in range(num_batch_per_epoch):
            #continue
            batch_t += 1
            image_t, text_t, label_dat_t = sess.run([image_bt_t, text_bt_t, label_bt_t])

            feed_dict = {tch_t.text_ph:text_t, tch_t.image_ph:image_t}
            label_tch_t, = sess.run([tch_t.labels], feed_dict=feed_dict)
            sample_t = utils.generate_label(flags, label_dat_t, label_tch_t)
            feed_dict = {
              dis_t.image_ph:image_t,
              dis_t.sample_ph:sample_t,
            }
            reward_t, = sess.run([dis_t.rewards], feed_dict=feed_dict)
            
            feed_dict = {
              gen_t.image_ph:image_t,
            }
            label_gen_g = sess.run(gen_t.logits, feed_dict = feed_dict)
            #print(len(label_dat_t), len(label_dat_t[0]))
            #exit()
            feed_dict = {
              tch_t.text_ph:text_t,
              tch_t.image_ph: image_t,
              tch_t.sample_ph:sample_t,
              tch_t.reward_ph:reward_t,
              tch_t.hard_label_ph:label_dat_t,
              tch_t.soft_label_ph:label_gen_g,
            }

            _, summary_t, tch_kdgan_loss = sess.run([tch_t.kdgan_update, tch_summary_op, tch_t.kdgan_loss], 
                feed_dict=feed_dict)
            writer.add_summary(summary_t, batch_t)
            #print("teacher kdgan loss:", tch_kdgan_loss)

        for gen_epoch in range(flags.num_gen_epoch):
          print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
          for _ in range(num_batch_per_epoch):
            batch_g += 1
            image_g, text_g, label_dat_g = sess.run([image_bt_g, text_bt_g, label_bt_g])

            feed_dict = {tch_t.text_ph:text_g, tch_t.image_ph: image_g}
            label_tch_g, = sess.run([tch_t.labels], feed_dict=feed_dict)
            # print('tch label {}'.format(label_tch_g.shape))

            feed_dict = {gen_t.image_ph:image_g}
            label_gen_g, = sess.run([gen_t.labels], feed_dict=feed_dict)
            sample_g = utils.generate_label(flags, label_dat_g, label_gen_g)
            feed_dict = {
              dis_t.image_ph:image_g,
              dis_t.sample_ph:sample_g,
            }
            reward_g, = sess.run([dis_t.rewards], feed_dict=feed_dict)

            feed_dict = {
              gen_t.image_ph:image_g,
              gen_t.hard_label_ph:label_dat_g,
              gen_t.soft_label_ph:label_tch_g,
              gen_t.sample_ph:sample_g,
              gen_t.reward_ph:reward_g,
            }
            _, summary_g = sess.run([gen_t.kdgan_update, gen_summary_op], 
                feed_dict=feed_dict)
            writer.add_summary(summary_g, batch_g)

            # if (batch_g + 1) % eval_interval != 0:
            #     continue
            # gen_hit = utils.evaluate_image(flags, sess, gen_v, bt_list_v)
            # tot_time = time.time() - start
            # print('#%08d hit=%.4f %06ds' % (batch_g, gen_hit, int(tot_time)))
            # if gen_hit <= best_hit_v:
            #   continue
            # best_hit_v = gen_hit
            # print('best hit=%.4f' % (best_hit_v))
        gen_hit = utils.evaluate_image(flags, sess, gen_v, bt_list_v)
        tch_hit = utils.evaluate_text(flags, sess, tch_v, bt_list_v)

        tot_time = time.time() - start
        print('#%03d curgen=%.4f curtch=%.4f %.0fs' % (epoch, gen_hit, tch_hit, tot_time))
        figure_data.append((epoch, gen_hit, tch_hit))
        if gen_hit <= best_hit_v:
          continue
        best_hit_v = gen_hit
        print("epoch ", epoch+1, ":, new best validation hit:", best_hit_v, "saving...")
        gen_t.saver.save(sess, flags.kdgan_model_ckpt, global_step= epoch+1)
        print("finish saving")

  print('best hit=%.4f' % (best_hit_v))
  

  utils.create_if_nonexist(os.path.dirname(flags.kdgan_figure_data))
  fout = open(flags.kdgan_figure_data, 'w')
  for epoch, gen_hit, tch_hit in figure_data:
    fout.write('%d\t%.4f\t%.4f\n' % (epoch, gen_hit, tch_hit))
  fout.close()

if __name__ == '__main__':
  tf.app.run()






