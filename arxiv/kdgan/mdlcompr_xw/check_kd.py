from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
from tch_model import TCH
import data_utils

from os import path
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import os
import time
import numpy as np
import tensorflow as tf

# dataset
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_integer('channels', 1, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('num_label', 10, '')
tf.app.flags.DEFINE_integer('train_size', 60000, '')
tf.app.flags.DEFINE_integer('valid_size', 0, '')
# model
tf.app.flags.DEFINE_float('gen_keep_prob', 0.95, '')
tf.app.flags.DEFINE_float('tch_keep_prob', 0.5, '')
tf.app.flags.DEFINE_float('kd_hard_pct', 0.3, '')
tf.app.flags.DEFINE_float('kd_soft_pct', 1.0, '')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_string('gen_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
# optimization
tf.app.flags.DEFINE_float('gen_weight_decay', 0.0, 'l2 coefficient')
tf.app.flags.DEFINE_float('gen_opt_epsilon', 1e-6, '')
tf.app.flags.DEFINE_float('tch_weight_decay', 0.0, 'l2 coefficient')
tf.app.flags.DEFINE_float('tch_opt_epsilon', 1e-6, '')
tf.app.flags.DEFINE_float('clip_norm', 10.0, '')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.0, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_integer('num_batch', 20, '')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'rmsprop|sgd')
# learning rate
tf.app.flags.DEFINE_float('gen_learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('gen_learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('gen_num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_float('tch_learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('tch_learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('tch_num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS

mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
print('tn size=%d vd size=%d' % (mnist.train.num_examples, mnist.test.num_examples))
tn_num_batch = int(flags.num_epoch * mnist.train.num_examples / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))

tn_tch = TCH(flags, mnist.train, is_training=True)
tn_gen = GEN(flags, mnist.train, is_training=True)
tn_sl_gen = GEN(flags, mnist.train, is_training=True, gen_scope='sl_gen')
tn_kd_gen = GEN(flags, mnist.train, is_training=True, gen_scope='kd_gen')
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_tch = TCH(flags, mnist.test, is_training=False)
vd_gen = GEN(flags, mnist.test, is_training=False)
vd_sl_gen = GEN(flags, mnist.test, is_training=False, gen_scope='sl_gen')
vd_kd_gen = GEN(flags, mnist.test, is_training=False, gen_scope='kd_gen')

learning_rate_diff = tn_sl_gen.learning_rate - tn_kd_gen.learning_rate
# sl_summary_op = tf.summary.merge([
#   tf.summary.scalar('accuracy', vd_sl_gen.accuracy)
# ])
# kd_summary_op = tf.summary.merge([
#   tf.summary.scalar('accuracy', vd_kd_gen.accuracy)
# ])
acc_diff = vd_kd_gen.accuracy - vd_sl_gen.accuracy
tf.summary.scalar('kd_minus_sl', acc_diff)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

init_sl_gen_ckpt = path.join(flags.gen_checkpoint_dir, 'sl_gen')
init_kd_gen_ckpt = path.join(flags.gen_checkpoint_dir, 'kd_gen')

def ini():
  start = time.time()
  with tf.Session() as sess:
    sess.run(init_op)
    ini_acc = metric.eval_mdlcompr(sess, vd_gen, mnist)
    # print('%-25s:%.4f' % ('ini', ini_acc))

    for var, sl_var, kd_var in zip(tn_gen.var_list, tn_sl_gen.var_list, tn_kd_gen.var_list):
      # print('%-50s\n%-50s\n%-50s' % (var.name, sl_var.name, kd_var.name))
      var_value = sess.run(var)
      sl_assign = sl_var.assign(var_value)
      sess.run(sl_assign)
      kd_assign = kd_var.assign(var_value)
      sess.run(kd_assign)

    # ini_sl_acc = metric.eval_mdlcompr(sess, vd_sl_gen, mnist)
    # ini_kd_acc = metric.eval_mdlcompr(sess, vd_kd_gen, mnist)
    # print('%-25s:%.4f\n%-25s:%.4f' % ('ini sl', ini_sl_acc, 'ini kd', ini_kd_acc))

    tn_sl_gen.saver.save(sess, init_sl_gen_ckpt)
    tn_kd_gen.saver.save(sess, init_kd_gen_ckpt)
  tot_time = time.time() - start

def run():
  count = 0
  best_sl_acc, best_kd_acc = 0.0, 0.0
  start = time.time()
  tch_model_ckpt = utils.get_latest_ckpt(flags.tch_checkpoint_dir)
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  # sl_writer = tf.summary.FileWriter(path.join(config.logs_dir, 'sl'), graph=tf.get_default_graph())
  # kd_writer = tf.summary.FileWriter(path.join(config.logs_dir, 'kd'), graph=tf.get_default_graph())
  fout = open(path.join(config.logs_dir, 'mdlcompr_check_kd.txt'), 'w')
  fout.write('%s\t%s\n' % ('sl', 'kd'))
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_tch.saver.restore(sess, tch_model_ckpt)
    tn_sl_gen.saver.restore(sess, init_sl_gen_ckpt)
    tn_kd_gen.saver.restore(sess, init_kd_gen_ckpt)
    ini_tch_acc = metric.eval_mdlcompr(sess, vd_tch, mnist)
    print('%-25s:%.4f' % ('ini tch', ini_tch_acc))

    ini_sl_acc = metric.eval_mdlcompr(sess, vd_sl_gen, mnist)
    ini_kd_acc = metric.eval_mdlcompr(sess, vd_kd_gen, mnist)
    print('%-25s:%.4f\n%-25s:%.4f' % ('ini sl', ini_sl_acc, 'ini kd', ini_kd_acc))

    # for tn_batch in range(tn_num_batch):
    for tn_batch in range(flags.num_batch):
      tn_images, tn_labels = mnist.train.next_batch(flags.batch_size)
      # print(tn_images.shape, tn_labels.shape)
      feed_dict = {vd_tch.image_ph:tn_images}
      soft_logits, = sess.run([vd_tch.logits], feed_dict=feed_dict)

      feed_dict = {
        tn_sl_gen.image_ph:tn_images,
        tn_sl_gen.hard_label_ph:tn_labels,
        tn_kd_gen.image_ph:tn_images,
        tn_kd_gen.hard_label_ph:tn_labels,
        tn_kd_gen.soft_logit_ph:soft_logits,
      }
      sess.run([tn_sl_gen.pre_update, tn_kd_gen.kd_update], feed_dict=feed_dict)

      # if tn_batch % 100 != 0:
      #   continue
      sl_acc = metric.eval_mdlcompr(sess, vd_sl_gen, mnist)
      kd_acc = metric.eval_mdlcompr(sess, vd_kd_gen, mnist)
      # feed_dict = {
      #   vd_sl_gen.image_ph:mnist.test.images, 
      #   vd_sl_gen.hard_label_ph:mnist.test.labels,
      #   vd_kd_gen.image_ph:mnist.test.images,
      #   vd_kd_gen.hard_label_ph:mnist.test.labels,
      # }
      # sl_gen_acc = sess.run(vd_sl_gen.accuracy, feed_dict=feed_dict)
      # kd_gen_acc = sess.run(vd_kd_gen.accuracy, feed_dict=feed_dict)
      # print(sl_gen_acc - sl_acc, kd_gen_acc - kd_acc)

      # sl_summary, kd_summary = sess.run([sl_summary_op, kd_summary_op],
      #     feed_dict=feed_dict)
      # sl_writer.add_summary(sl_summary, tn_batch)
      # kd_writer.add_summary(kd_summary, tn_batch)
      # summary = sess.run(summary_op, feed_dict=feed_dict)
      # writer.add_summary(summary, tn_batch)

      # if kd_gen_acc < sl_gen_acc:
      #   continue
      print('%d %.4f %.4f' % (tn_batch + 1, sl_acc, kd_acc))
      fout.write('%d\t%.4f\t%.4f\n' % (tn_batch + 1, sl_acc, kd_acc))
      # count += 1
      # if best_sl_acc < sl_gen_acc:
      #   best_sl_acc = sl_gen_acc
      # if best_kd_acc < kd_gen_acc:
      #   best_kd_acc = kd_gen_acc
      # print('%d %.4f %.4f' % (tn_batch + 1, best_sl_acc, best_kd_acc))
      if (tn_batch % 1000) != 0:
        continue
      fout.flush()

  tot_time = time.time() - start
  # print('cn=%d tm=%.0fs' % (count, tot_time))
  # print('bst %.4f %.4f' % (best_sl_acc, best_kd_acc))
  fout.close()

def main(_):
  ini()
  run()

if __name__ == '__main__':
    tf.app.run()









