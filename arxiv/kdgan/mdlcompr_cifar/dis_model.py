from kdgan import utils
import lenet_utils

import numpy as np
import tensorflow as tf

class DIS():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32,
        shape=(flags.batch_size, flags.image_size, flags.image_size, flags.channels))
    self.hard_label_ph = tf.placeholder(tf.int32,
        shape=(flags.batch_size, flags.num_label))

    # None = batch_size * sample_size
    self.std_sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.std_label_ph = tf.placeholder(tf.float32, shape=(None,))
    self.tch_sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.tch_label_ph = tf.placeholder(tf.float32, shape=(None,))

    self.dis_scope = dis_scope = 'dis'
    with tf.variable_scope(dis_scope):
      self.logits = lenet_utils.inference(self.image_ph)
      self.labels = tf.nn.softmax(self.logits)

      self.std_rewards = self.get_rewards(self.std_sample_ph)
      # self.tch_rewards = self.get_rewards(self.tch_sample_ph)

      if not is_training:
        predictions = tf.argmax(self.labels, axis=1)
        groundtruth = tf.argmax(self.hard_label_ph, axis=1)
        accuracy_list = tf.equal(predictions, groundtruth)
        self.accuracy = tf.reduce_mean(tf.cast(accuracy_list, tf.float32))
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(dis_scope):
          continue
        print('%-64s added to DIS saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)

      self.global_step = global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.Variable(flags.dis_learning_rate, trainable=False)

      # pre train
      pre_losses = self.get_pre_losses()
      print('#pre_losses wo regularization=%d' % (len(pre_losses)))
      pre_losses.extend(self.get_regularization_losses())
      print('#pre_losses wt regularization=%d' % (len(pre_losses)))
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % dis_scope)
      self.pre_train = lenet_utils.get_train_op(self.pre_loss, global_step)

      # gan train
      gan_losses = self.get_gan_losses(flags)
      print('#gan_losses wo regularization=%d' % (len(gan_losses)))
      gan_losses.extend(self.get_regularization_losses())
      print('#gan_losses wt regularization=%d' % (len(gan_losses)))
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % dis_scope)
      gan_optimizer = utils.get_opt(flags, self.learning_rate)
      # gan_optimizer = tf.train.AdamOptimizer(self.learning_rate)
      # gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.gan_train = gan_optimizer.minimize(self.gan_loss, global_step=self.global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.dis_scope):
        continue
      print('DIS regularization=%s' % (regularization_loss.name))
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_hard_loss(self):
    hard_loss = lenet_utils.loss(self.logits, self.hard_label_ph)
    return hard_loss

  def get_pre_losses(self):
    pre_losses = [self.get_hard_loss()]
    return pre_losses

  def get_gan_loss(self, sample_ph, label_ph):
    sample_logits = tf.gather_nd(self.logits, sample_ph)
    gan_loss = tf.losses.sigmoid_cross_entropy(label_ph, sample_logits)
    return gan_loss

  def get_gan_losses(self, flags):
    std_gan_loss = self.get_gan_loss(self.std_sample_ph, self.std_label_ph)
    std_gan_loss *= (1.0 - flags.intelltch_weight)
    gan_losses = [std_gan_loss]
    # tch_gan_loss = self.get_gan_loss(self.tch_sample_ph, self.tch_label_ph)
    # tch_gan_loss *= flags.intelltch_weight
    # gan_losses.append(tch_gan_loss)
    return gan_losses

  def get_rewards(self, sample_ph):
    reward_logits = tf.sigmoid(self.logits)
    rewards = tf.gather_nd(reward_logits, sample_ph)
    return rewards











