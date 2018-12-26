from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from data_utils import AffineGenerator
from gen_model import GEN
from tch_model import TCH
import data_utils

from os import path
from tensorflow.contrib import slim
import math
import os
import time
import numpy as np
import tensorflow as tf

mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
datagen = AffineGenerator(mnist)

tn_size, vd_size = mnist.train.num_examples, mnist.test.num_examples
print('tn size=%d vd size=%d' % (tn_size, vd_size))
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(tn_size / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_gen = GEN(flags, mnist.train, is_training=True)
tn_tch = TCH(flags, mnist.train, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, mnist.test, is_training=False)
vd_tch = TCH(flags, mnist.test, is_training=False)

tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.kd_loss.name, tn_gen.kd_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

tot_params = 0
for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))
  tot_params += num_params
print('%-50s (%d params)' % (' '.join(['kd', flags.kd_model]), tot_params))

def main(_):
  bst_acc, bst_epk = 0.0, 0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)
    tn_tch.saver.restore(sess, flags.tch_model_ckpt)
    ini_gen = metric.eval_mdlcompr(sess, vd_gen, mnist)
    ini_tch = metric.eval_mdlcompr(sess, vd_tch, mnist)
    # print('ini_gen=%.4f ini_tch=%.4f' % (ini_gen, ini_tch))
    fout = open('std-tch.txt', 'a+')
    fout.write('%05d\t%.4f\t%.4f\n' % (flags.train_size, ini_gen, ini_tch))
    fout.close()
    exit()

    start = time.time()
    # for tn_batch in range(tn_num_batch):
    #   tn_image_np, tn_label_np = mnist.train.next_batch(flags.batch_size)
    tn_batch = -1
    for epoch in range(flags.num_epoch):
      for tn_image_np, tn_label_np in datagen.generate(batch_size=flags.batch_size):
        tn_batch += 1
        feed_dict = {vd_tch.image_ph:tn_image_np}
        soft_logit_np, = sess.run([vd_tch.logits], feed_dict=feed_dict)

        feed_dict = {
          tn_gen.image_ph:tn_image_np,
          tn_gen.hard_label_ph:tn_label_np,
          tn_gen.soft_logit_ph:soft_logit_np,
        }
        _, summary = sess.run([tn_gen.kd_update, summary_op], feed_dict=feed_dict)
        writer.add_summary(summary, tn_batch)

        if (tn_batch + 1) % eval_interval != 0:
          continue
        feed_dict = {
          vd_gen.image_ph:mnist.test.images,
          vd_gen.hard_label_ph:mnist.test.labels,
        }
        acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)

        if acc > bst_acc:
          bst_acc = max(acc, bst_acc)
          bst_epk = epoch
        tot_time = time.time() - start
        global_step = sess.run(tn_gen.global_step)
        avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
        print('#%08d curacc=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
            (tn_batch, acc, bst_acc, tot_time, avg_time))

        if acc <= bst_acc:
          continue
        # save gen parameters if necessary
  tot_time = time.time() - start
  ini_gen *= 100
  bst_acc *= 100
  bst_epk += 1
  print('#mnist=%d %s@%d=%.2f iniacc=%.2f et=%.0fs' % 
      (tn_size, flags.kd_model, bst_epk, bst_acc, ini_gen, tot_time))

if __name__ == '__main__':
    tf.app.run()









