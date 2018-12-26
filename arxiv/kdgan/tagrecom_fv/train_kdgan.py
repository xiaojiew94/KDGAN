from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from dis_model import DIS
from gen_model import GEN
from tch_model import TCH
import data_utils

import math
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import slim

# print('alpha=%.8f beta=%.8f gamma=%.8f' % (flags.intelltch_weight, flags.distilled_weight, flags.intellstd_weight))

tn_size = utils.get_tn_size(flags.dataset)
eval_interval = int(tn_size / flags.batch_size)
print('#tn_size=%d' % (tn_size))

tn_dis = DIS(flags, is_training=True)
tn_gen = GEN(flags, is_training=True)
tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, is_training=False)
vd_gen = GEN(flags, is_training=False)
vd_tch = TCH(flags, is_training=False)

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

dis_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_dis.learning_rate.name, tn_dis.learning_rate),
  tf.summary.scalar(tn_dis.gan_loss.name, tn_dis.gan_loss),
])
gen_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate),
  tf.summary.scalar(tn_gen.kdgan_loss.name, tn_gen.kdgan_loss),
])
tch_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_tch.learning_rate.name, tn_tch.learning_rate),
  tf.summary.scalar(tn_tch.kdgan_loss.name, tn_tch.kdgan_loss),
])
init_op = tf.global_variables_initializer()

yfccdata_d = data_utils.YFCCDATA(flags)
yfccdata_g = data_utils.YFCCDATA(flags)
yfccdata_t = data_utils.YFCCDATA(flags)
yfcceval = data_utils.YFCCEVAL(flags)

def main(_):
  best_prec, bst_epk = 0.0, 0
  epk_score_list = []
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_dis.saver.restore(sess, flags.dis_model_ckpt)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)
    tn_tch.saver.restore(sess, flags.tch_model_ckpt)
    start = time.time()

    ini_dis = yfcceval.compute_prec(flags, sess, vd_dis)
    ini_gen = yfcceval.compute_prec(flags, sess, vd_gen)
    ini_tch = yfcceval.compute_prec(flags, sess, vd_tch)
    print('ini dis=%.4f gen=%.4f tch=%.4f' % (ini_dis, ini_gen, ini_tch))

    batch_d, batch_g, batch_t = -1, -1, -1
    for epoch in range(flags.num_epoch):
      num_batch_d = math.ceil(flags.num_dis_epoch * tn_size / flags.batch_size)
      for _ in range(num_batch_d):
        batch_d += 1
        image_d, text_d, label_dat_d = yfccdata_d.next_batch(flags, sess)
        
        feed_dict = {tn_gen.image_ph:image_d}
        label_gen_d = sess.run(tn_gen.labels, feed_dict=feed_dict)
        sample_gen_d, gen_label_d = utils.gan_dis_sample(flags, label_dat_d, label_gen_d)

        feed_dict = {tn_tch.image_ph:image_d, tn_tch.text_ph:text_d}
        label_tch_d = sess.run(tn_tch.labels, feed_dict=feed_dict)
        sample_tch_d, tch_label_d = utils.gan_dis_sample(flags, label_dat_d, label_tch_d)

        feed_dict = {
          tn_dis.image_ph:image_d,
          tn_dis.gen_sample_ph:sample_gen_d,
          tn_dis.gen_label_ph:gen_label_d,
          tn_dis.tch_sample_ph:sample_tch_d,
          tn_dis.tch_label_ph:tch_label_d,
        }
        _, summary_d = sess.run([tn_dis.gan_update, dis_summary_op], feed_dict=feed_dict)
        writer.add_summary(summary_d, batch_d)

      num_batch_t = math.ceil(flags.num_tch_epoch * tn_size / flags.batch_size)
      for _ in range(num_batch_t):
        batch_t += 1
        image_t, text_t, label_dat_t = yfccdata_t.next_batch(flags, sess)

        feed_dict = {tn_tch.image_ph:image_t, tn_tch.text_ph:text_t}
        label_tch_t = sess.run(tn_tch.labels, feed_dict=feed_dict)
        sample_t = utils.generate_label(flags, label_dat_t, label_tch_t)
        feed_dict = {tn_dis.image_ph:image_t, tn_dis.tch_sample_ph:sample_t}
        reward_t = sess.run(tn_dis.tch_rewards, feed_dict=feed_dict)

        feed_dict = {vd_gen.image_ph:image_t}
        soft_logit_t = sess.run(vd_gen.logits, feed_dict=feed_dict)

        feed_dict = {
          tn_tch.image_ph:image_t,
          tn_tch.text_ph:text_t,
          tn_tch.sample_ph:sample_t,
          tn_tch.reward_ph:reward_t,
          tn_tch.hard_label_ph:label_dat_t,
          tn_tch.soft_logit_ph:soft_logit_t,
        }
        _, summary_t = sess.run([tn_tch.kdgan_update, tch_summary_op], feed_dict=feed_dict)
        writer.add_summary(summary_t, batch_t)

      num_batch_g = math.ceil(flags.num_gen_epoch * tn_size / flags.batch_size)
      for _ in range(num_batch_g):
        batch_g += 1
        image_g, text_g, label_dat_g = yfccdata_g.next_batch(flags, sess)

        feed_dict = {tn_tch.image_ph:image_g, tn_tch.text_ph:text_g}
        logit_tch_g = sess.run(tn_tch.logits, feed_dict=feed_dict)
        # print('tch label {}'.format(logit_tch_g.shape))

        feed_dict = {tn_gen.image_ph:image_g}
        label_gen_g = sess.run(tn_gen.labels, feed_dict=feed_dict)
        sample_g = utils.generate_label(flags, label_dat_g, label_gen_g)
        feed_dict = {tn_dis.image_ph:image_g, tn_dis.gen_sample_ph:sample_g}
        reward_g = sess.run(tn_dis.gen_rewards, feed_dict=feed_dict)

        feed_dict = {
          tn_gen.image_ph:image_g,
          tn_gen.hard_label_ph:label_dat_g,
          tn_gen.soft_logit_ph:logit_tch_g,
          tn_gen.sample_ph:sample_g,
          tn_gen.reward_ph:reward_g,
        }
        _, summary_g = sess.run([tn_gen.kdgan_update, gen_summary_op], feed_dict=feed_dict)
        writer.add_summary(summary_g, batch_g)

        if (batch_g + 1) % eval_interval != 0:
            continue
        scores = yfcceval.compute_score(flags, sess, vd_gen)
        epk_score_list.append(scores)
        p3, p5, f3, f5, ndcg3, ndcg5, ap, rr = scores
        print('p3=%.4f p5=%.4f f3=%.4f f5=%.4f ndcg3=%.4f ndcg5=%.4f ap=%.4f rr=%.4f' % 
            (p3, p5, f3, f5, ndcg3, ndcg5, ap, rr))
        prec = yfcceval.compute_prec(flags, sess, vd_gen)
        if prec > best_prec:
          bst_epk = epoch
        best_prec = max(prec, best_prec)
        tot_time = time.time() - start
        global_step = sess.run(tn_gen.global_step)
        avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
        print('#%08d@%d prec@%d=%.4f best@%d=%.4f tot=%.0fs avg=%.2fs/epoch' % 
            (global_step, epoch, flags.cutoff, prec, bst_epk, best_prec, tot_time, avg_time))

        if prec < best_prec:
          continue
        # save if necessary
  tot_time = time.time() - start
  print('best@%d=%.4f et=%.0fs' % (flags.cutoff, best_prec, tot_time))

  if flags.epk_learning_curve_p == None:
    return
  utils.create_pardir(flags.epk_learning_curve_p)
  pickle.dump(epk_score_list, open(flags.epk_learning_curve_p, 'wb'))

if __name__ == '__main__':
  tf.app.run()





