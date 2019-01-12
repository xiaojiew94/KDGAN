from kdgan import config, utils

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
    # None = batch_size * (num_positive + num_negative)
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    # None = batch_size * (num_positive + num_negative)
    self.label_ph = tf.placeholder(tf.float32, shape=(None,))
    # None = batch_size
    self.pre_label_ph = tf.placeholder(tf.float32, shape=(None, config.num_label))

    dis_scope = 'discriminator'
    model_scope = nets_factory.arg_scopes_map[flags.model_name]
    with tf.variable_scope(dis_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.dis_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.dropout_keep_prob, 
            is_training=is_training)
        net = slim.fully_connected(net, config.num_label,
            activation_fn=None)
        self.logits = net

    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    # self.rewards = 2 * (tf.sigmoid(sample_logits) - 0.5)
    # self.rewards = tf.sigmoid(sample_logits)

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
      print('%s added to DIS saver' % variable.name)
      save_dict[variable.name] = variable
    self.saver = tf.train.Saver(save_dict)

    train_data_size = utils.get_tn_size(flags.dataset)
    global_step = tf.train.get_global_step()
    decay_steps = int(train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
    self.learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
        global_step, decay_steps, flags.learning_rate_decay_factor,
        staircase=True, name='exponential_decay_learning_rate')

    # pretrain discriminator
    losses = []
    losses.append(tf.losses.sigmoid_cross_entropy(self.pre_label_ph, self.logits))
    losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.pre_loss = tf.add_n(losses, name='dis_pre_loss')
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.pre_train_op = optimizer.minimize(self.pre_loss, global_step=global_step)

    losses = []
    losses.append(tf.losses.sigmoid_cross_entropy(self.label_ph, sample_logits))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    losses.extend(regularization_losses)
    self.gan_loss = tf.add_n(losses, name='dis_gan_loss')
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_op = optimizer.minimize(self.gan_loss, global_step=global_step)


