from kdgan import config
from flags import flags
import data_utils
import resnet_utils

import numpy as np
import tensorflow as tf
import math
import six
import sys
import time

def train(hps):
  """Training loop."""
  images, labels = data_utils.build_input(
      flags.dataset, flags.train_data_path, hps.batch_size, flags.mode)
  model = resnet_utils.ResNet(hps, images, labels, flags.mode)
  model.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=config.logs_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 60000:
        self._lrn_rate = 0.01
      elif train_step < 80000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=flags.tch_ckpt_dir,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)


def valid(hps):
  vd_image_ts, vd_label_ts = data_utils.build_input(flags, 'valid')
  image_ph = tf.placeholder(tf.float32, 
      shape=(flags.batch_size, flags.image_size, flags.image_size, flags.channels))
  hard_label_ph = tf.placeholder(tf.float32, 
      shape=(flags.batch_size, flags.num_label))

  model = resnet_utils.ResNet(hps, image_ph, hard_label_ph, 'valid')
  model.build_graph()
  saver = tf.train.Saver()

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  try:
    ckpt_state = tf.train.get_checkpoint_state(flags.tch_ckpt_dir)
  except tf.errors.OutOfRangeError as e:
    tf.logging.error('cannot restore checkpoint: %s', e)
    return
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('no model to eval yet at %s', flags.tch_ckpt_dir)
    return
  tf.logging.info('loading checkpoint %s', ckpt_state.model_checkpoint_path)
  saver.restore(sess, ckpt_state.model_checkpoint_path)

  vd_num_batch = int(math.ceil(flags.valid_size / flags.batch_size))
  total_prediction, correct_prediction = 0, 0
  for _ in range(vd_num_batch):
    vd_image_np, vd_label_np = sess.run([vd_image_ts, vd_label_ts])
    feed_dict = {image_ph:vd_image_np, hard_label_ph:vd_label_np}
    predictions = sess.run(model.predictions, feed_dict=feed_dict)

    vd_label_np = np.argmax(vd_label_np, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(vd_label_np == predictions)
    total_prediction += predictions.shape[0]
    print(correct_prediction, total_prediction)
  acc = 1.0 * correct_prediction / total_prediction
  tf.logging.info('acc=%.4f' % (acc))

def dev(hps):
  images, labels = data_utils.build_input(flags, 'valid')
  model = resnet_utils.ResNet(hps, images, labels, 'valid')
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(config.logs_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  try:
    ckpt_state = tf.train.get_checkpoint_state(flags.tch_ckpt_dir)
  except tf.errors.OutOfRangeError as e:
    tf.logging.error('cannot restore checkpoint: %s', e)
    return
  if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('no model to eval yet at %s', flags.tch_ckpt_dir)
    return
  tf.logging.info('loading checkpoint %s', ckpt_state.model_checkpoint_path)
  saver.restore(sess, ckpt_state.model_checkpoint_path)

  vd_num_batch = int(math.ceil(flags.valid_size / flags.batch_size))
  total_prediction, correct_prediction = 0, 0
  for _ in six.moves.range(vd_num_batch):
    (summaries, loss, predictions, truth, train_step) = sess.run(
        [model.summaries, model.cost, model.predictions,
         model.labels, model.global_step])

    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]

  precision = 1.0 * correct_prediction / total_prediction

  precision_summ = tf.Summary()
  precision_summ.value.add(tag='Precision', simple_value=precision)
  summary_writer.add_summary(precision_summ, train_step)
  summary_writer.add_summary(summaries, train_step)
  tf.logging.info('precision: %.4f' % (precision))
  summary_writer.flush()

def main(_):
  hps = resnet_utils.HParams(batch_size=flags.batch_size,
                             num_classes=flags.num_label,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             # num_residual_units=5,
                             num_residual_units=3,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  # train(hps)
  # valid(hps)
  dev(hps)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
