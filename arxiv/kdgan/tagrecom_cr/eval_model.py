from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS
from gen_model import GEN
from tch_model import TCH
import flags

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('model_ckpt', None, '')
tf.app.flags.DEFINE_string('checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('model_run', None, '')
flags = tf.app.flags.FLAGS

model = None
if flags.model_name == 'dis':
  model = DIS
elif flags.model_name == 'gen':
  model = GEN
elif flags.model_name == 'tch':
  model = TCH
else:
  raise ValueError('bad model name %s' % flags.model_name)

tn_model = model(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_model = model(flags, is_training=False)

for var in tf.trainable_variables():
  num_params = 1
  for dim in var.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (var.name, num_params))

image_np, text_np, label_np, imgid_np = utils.get_valid_data(flags)

model_ckpt = flags.model_ckpt
if flags.checkpoint_dir != None:
  model_ckpt = utils.get_latest_ckpt(flags.checkpoint_dir)

def main(_):
  utils.create_pardir(flags.model_run)
  id_to_label = utils.load_id_to_label(flags.dataset)
  fout = open(flags.model_run, 'w')
  with tf.train.MonitoredTrainingSession() as sess:
    tn_model.saver.restore(sess, model_ckpt)
    if hasattr(vd_model, 'text_ph'):
      feed_dict = {
        vd_model.image_ph:image_np,
        vd_model.text_ph:text_np,
      }
    else:
      feed_dict = {
        vd_model.image_ph:image_np,
      }
    logit_np = sess.run(vd_model.logits, feed_dict=feed_dict)
    for imgid, logit_np in zip(imgid_np, logit_np):
      sorted_labels = (-logit_np).argsort()
      fout.write('%s' % (imgid))
      for label in sorted_labels:
        fout.write(' %s %.4f' % (id_to_label[label], logit_np[label]))
      fout.write('\n')
  fout.close()
  print('result saved in %s' % flags.model_run)

if __name__ == '__main__':
  tf.app.run()