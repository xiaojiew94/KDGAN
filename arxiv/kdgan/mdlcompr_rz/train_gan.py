from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from data_utils import AffineGenerator
from dis_model import DIS
from gen_model import GEN
import data_utils

import math
import os
import pickle
import time
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import slim

dis_mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
dis_datagen = AffineGenerator(dis_mnist)
gen_mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
gen_datagen = AffineGenerator(gen_mnist)

tn_size = int((dis_mnist.train.num_examples + gen_mnist.train.num_examples) / 2)
vd_size = int((dis_mnist.test.num_examples + gen_mnist.test.num_examples) / 2)
print('tn size=%d vd size=%d' % (tn_size, vd_size))
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(tn_size / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_dis = DIS(flags, dis_mnist.train, is_training=True)
tn_gen = GEN(flags, gen_mnist.train, is_training=True)
dis_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_dis.learning_rate.name, tn_dis.learning_rate),
  tf.summary.scalar(tn_dis.gan_loss.name, tn_dis.gan_loss),
])
gen_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate),
  tf.summary.scalar(tn_gen.gan_loss.name, tn_gen.gan_loss),
])
init_op = tf.global_variables_initializer()

scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, dis_mnist.test, is_training=False)
vd_gen = GEN(flags, gen_mnist.test, is_training=False)

# for variable in tf.trainable_variables():
#   num_params = 1
#   for dim in variable.shape:
#     num_params *= dim.value
#   print('%-50s (%d params)' % (variable.name, num_params))

def main(_):
  bst_acc = 0.0
  acc_list = []
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_dis.saver.restore(sess, flags.dis_model_ckpt)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)

    feed_dict = {
      vd_dis.image_ph:dis_mnist.test.images,
      vd_dis.hard_label_ph:dis_mnist.test.labels,
    }
    ini_dis = sess.run(vd_dis.accuracy, feed_dict=feed_dict)
    feed_dict = {
      vd_gen.image_ph:gen_mnist.test.images,
      vd_gen.hard_label_ph:gen_mnist.test.labels,
    }
    ini_gen = sess.run(vd_gen.accuracy, feed_dict=feed_dict)
    print('ini dis=%.4f ini gen=%.4f' % (ini_dis, ini_gen))
    # exit()

    start = time.time()
    batch_d, batch_g = -1, -1
    for epoch in range(flags.num_epoch):
      for dis_epoch in range(flags.num_dis_epoch):
        # print('epoch %03d dis_epoch %03d' % (epoch, dis_epoch))
        num_batch_d = math.ceil(tn_size / flags.batch_size)
        for image_np_d, label_dat_d in dis_datagen.generate(batch_size=flags.batch_size):
        # for _ in range(num_batch_d):
        #   image_np_d, label_dat_d = dis_mnist.train.next_batch(flags.batch_size)
          batch_d += 1
          feed_dict = {tn_gen.image_ph:image_np_d}
          label_gen_d, = sess.run([tn_gen.labels], feed_dict=feed_dict)
          # print('label_dat_d={} label_gen_d={}'.format(label_dat_d.shape, label_gen_d.shape))
          sample_np_d, label_np_d = utils.gan_dis_sample_dev(flags, label_dat_d, label_gen_d)
          feed_dict = {
            tn_dis.image_ph:image_np_d,
            tn_dis.sample_ph:sample_np_d,
            tn_dis.dis_label_ph:label_np_d,
          }
          _, summary_d = sess.run([tn_dis.gan_update, dis_summary_op], feed_dict=feed_dict)
          writer.add_summary(summary_d, batch_d)

      for gen_epoch in range(flags.num_gen_epoch):
        # print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
        num_batch_g = math.ceil(tn_size / flags.batch_size)
        for image_np_g, label_dat_g in gen_datagen.generate(batch_size=flags.batch_size):
        # for _ in range(num_batch_g):
        #   image_np_g, label_dat_g = gen_mnist.train.next_batch(flags.batch_size)
          batch_g += 1
          feed_dict = {tn_gen.image_ph:image_np_g}
          label_gen_g, = sess.run([tn_gen.labels], feed_dict=feed_dict)
          sample_np_g = utils.generate_label(flags, label_dat_g, label_gen_g)
          # sample_np_g, rescale_np_g = utils.generate_label(flags, label_dat_g, label_gen_g)
          # print(sample_np_g.shape, rescale_np_g.shape)
          feed_dict = {
            tn_dis.image_ph:image_np_g,
            tn_dis.sample_ph:sample_np_g,
          }
          reward_np_g, = sess.run([tn_dis.rewards], feed_dict=feed_dict)
          # reward_np_g *= rescale_np_g
          # print(reward_np_g)
          feed_dict = {
            tn_gen.image_ph:image_np_g,
            tn_gen.sample_ph:sample_np_g,
            tn_gen.reward_ph:reward_np_g,
          }
          _, summary_g = sess.run([tn_gen.gan_update, gen_summary_op], feed_dict=feed_dict)
          writer.add_summary(summary_g, batch_g)


          if flags.collect_cr_data:
            feed_dict = {
              vd_gen.image_ph:gen_mnist.test.images,
              vd_gen.hard_label_ph:gen_mnist.test.labels,
            }
            acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)
            acc_list.append(acc)
            if (batch_g + 1) % eval_interval != 0:
              continue
          else:
            if (batch_g + 1) % eval_interval != 0:
              continue
            feed_dict = {
              vd_gen.image_ph:gen_mnist.test.images,
              vd_gen.hard_label_ph:gen_mnist.test.labels,
            }
            acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)

          bst_acc = max(acc, bst_acc)
          tot_time = time.time() - start
          global_step = sess.run(tn_gen.global_step)
          avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
          print('#%08d curacc=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
              (batch_g, acc, bst_acc, tot_time, avg_time))

          if acc <= bst_acc:
            continue
          # save gen parameters if necessary
  tot_time = time.time() - start
  print('#mnist=%d bstacc=%.4f et=%.0fs' % (tn_size, bst_acc, tot_time))

  if flags.collect_cr_data:
    utils.create_pardir(flags.all_learning_curve_p)
    pickle.dump(acc_list, open(flags.all_learning_curve_p, 'wb'))

if __name__ == '__main__':
    tf.app.run()









