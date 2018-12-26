from kdgan import config, utils

from nets import nets_factory
from nets import vgg

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class GEN():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training
    
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size))
    self.label_ph = tf.placeholder(tf.float32, shape=(None, config.num_label))

    gen_scope = 'generator'
    model_scope = nets_factory.arg_scopes_map[flags.model_name]
    with tf.variable_scope(gen_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.gen_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.dropout_keep_prob, 
            is_training=is_training)
        net = slim.fully_connected(net, config.num_label,
            activation_fn=None)
        self.logits = net

    self.labels = tf.nn.softmax(self.logits)

    if not is_training:
      return

    save_dict = {}
    for variable in tf.trainable_variables():
      if not variable.name.startswith(gen_scope):
        continue
      print('%s added to GEN saver' % variable.name)
      save_dict[variable.name] = variable
    self.saver = tf.train.Saver(save_dict)

    train_data_size = utils.get_tn_size(flags.dataset)
    global_step = tf.train.get_global_step()
    decay_steps = int(train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
    self.learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
        global_step, decay_steps, flags.learning_rate_decay_factor,
        staircase=True, name='exponential_decay_learning_rate')

    # pretrain generator
    losses = []
    losses.append(tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    losses.extend(regularization_losses)
    self.pre_loss = tf.add_n(losses, name='pre_loss')
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_op = optimizer.minimize(self.pre_loss, global_step=global_step)

    # knowledge distillation
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, config.num_label))
    self.soft_label_ph = tf.placeholder(tf.float32, shape=(None, config.num_label))
    hard_loss = tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits)
    soft_loss = tf.nn.l2_loss(tf.nn.softmax(self.logits) - 
        tf.nn.softmax(self.soft_label_ph / flags.temperature) )
    kd_loss = (1.0 - flags.beta) * hard_loss + flags.beta * soft_loss
    kd_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.kd_train_op = kd_optimizer.minimize(kd_loss, global_step=global_step)

    # generative adversarial network
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    # gan_loss = -tf.reduce_mean(self.reward_ph * sample_logits)
    gan_loss = tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)
    gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.gan_train_op = gan_optimizer.minimize(gan_loss, global_step=global_step)

    # tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('diff', diff)
    # tf.summary.scalar('kd_loss', kd_loss)
    # tf.summary.scalar('gan_loss', gan_loss)
    # self.summary_op = tf.summary.merge_all()




