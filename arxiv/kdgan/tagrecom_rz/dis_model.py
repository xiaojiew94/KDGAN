from kdgan import config
from kdgan import utils

from nets import nets_factory
from nets import vgg

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class DIS():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training

    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.gen_sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.gen_label_ph = tf.placeholder(tf.float32, shape=(None,))
    self.tch_sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.tch_label_ph = tf.placeholder(tf.float32, shape=(None,))

    self.dis_scope = dis_scope = 'dis'
    model_scope = nets_factory.arg_scopes_map[flags.image_model]
    with tf.variable_scope(dis_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.image_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.image_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, flags.num_label, activation_fn=None)
        self.logits = net

      self.gen_rewards = self.get_rewards(self.gen_sample_ph)
      self.tch_rewards = self.get_rewards(self.tch_sample_ph)

      if not is_training:
        return

      save_dict = {}
      for variable in tf.trainable_variables():
        if not variable.name.startswith(dis_scope):
          continue
        print('%-50s added to DIS saver' % variable.name)
        save_dict[variable.name] = variable
      self.saver = tf.train.Saver(save_dict)

      self.global_step = global_step = tf.Variable(0, trainable=False)
      tn_size = utils.get_tn_size(flags.dataset)
      learning_rate = flags.dis_learning_rate
      self.learning_rate = utils.get_lr(flags, tn_size, global_step, learning_rate, dis_scope)

      # pre train
      pre_losses = self.get_pre_losses()
      pre_losses.extend(self.get_regularization_losses())
      print('#pre_losses wt regularization=%d' % (len(pre_losses)))
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % dis_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate)
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

      # gan train
      gan_losses = self.get_gan_losses(flags)
      gan_losses.extend(self.get_regularization_losses())
      print('#gan_losses wt regularization=%d' % (len(gan_losses)))
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % dis_scope)
      gan_optimizer = utils.get_opt(flags, self.learning_rate)
      self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=global_step)

  def get_hard_loss(self):
    hard_loss = tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits)
    return hard_loss

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.dis_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_pre_losses(self):
    pre_losses = [self.get_hard_loss()]
    # print('#pre_losses wo regularization=%d' % (len(pre_losses)))
    return pre_losses

  def get_gan_loss(self, sample_ph, label_ph):
    sample_logits = tf.gather_nd(self.logits, sample_ph)
    gan_loss = tf.losses.sigmoid_cross_entropy(label_ph, sample_logits)
    return gan_loss

  def get_gan_losses(self, flags):
    gen_gan_loss = self.get_gan_loss(self.gen_sample_ph, self.gen_label_ph)
    gen_gan_loss *= (1.0 - flags.alpha)
    tch_gan_loss = self.get_gan_loss(self.tch_sample_ph, self.tch_label_ph)
    tch_gan_loss *= flags.alpha
    gan_losses = [gen_gan_loss, tch_gan_loss]
    # print('#gan_losses wo regularization=%d' % (len(gan_losses)))
    return gan_losses

  def get_rewards(self, sample_ph):
    reward_logits = tf.sigmoid(self.logits)
    rewards = tf.gather_nd(reward_logits, sample_ph)
    return rewards

