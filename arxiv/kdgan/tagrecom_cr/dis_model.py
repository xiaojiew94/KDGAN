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
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.dis_label_ph = tf.placeholder(tf.float32, shape=(None,))

    dis_scope = 'dis'
    model_scope = nets_factory.arg_scopes_map[flags.image_model]
    with tf.variable_scope(dis_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.image_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.image_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, flags.num_label, activation_fn=None)
        self.logits = net

    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    reward_logits = self.logits
    # reward_logits = 2 * (tf.sigmoid(reward_logits) - 0.5)
    # reward_logits -= tf.reduce_mean(reward_logits, 1, keep_dims=True)
    # reward_logits -= tf.reduce_mean(reward_logits, 1, keep_dims=True)
    # reward_logits = 2 * (tf.sigmoid(reward_logits) - 0.5)
    reward_logits = tf.sigmoid(reward_logits)
    # reward_logits -= tf.reduce_mean(reward_logits, 1, keep_dims=True)
    self.rewards = tf.gather_nd(reward_logits, self.sample_ph)

    if not is_training:
      return

    save_dict = {}
    for variable in tf.trainable_variables():
      if not variable.name.startswith(dis_scope):
        continue
      print('%-50s added to DIS saver' % variable.name)
      save_dict[variable.name] = variable
    self.saver = tf.train.Saver(save_dict)

    global_step = tf.Variable(0, trainable=False)
    tn_size = utils.get_tn_size(flags.dataset)
    self.learning_rate = utils.get_lr(flags, 
        tn_size,
        global_step,
        flags.dis_learning_rate,
        dis_scope)
    
    # pre train
    pre_losses = []
    pre_losses.append(tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits))
    pre_losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % dis_scope)
    pre_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

    # gan train
    gan_losses = []
    gan_losses.append(tf.losses.sigmoid_cross_entropy(self.dis_label_ph, sample_logits))
    gan_losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % dis_scope)
    gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=global_step)


