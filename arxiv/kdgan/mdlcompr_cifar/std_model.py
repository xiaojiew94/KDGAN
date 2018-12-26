from kdgan import utils
import lenet_utils

import numpy as np
import tensorflow as tf

class STD():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32,
        shape=(flags.batch_size, flags.image_size, flags.image_size, flags.channels))
    self.hard_label_ph = tf.placeholder(tf.int32,
        shape=(flags.batch_size, flags.num_label))
    self.soft_logit_ph = tf.placeholder(tf.float32, 
        shape=(flags.batch_size, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.std_scope = std_scope = 'std'
    with tf.variable_scope(std_scope) as scope:
      self.logits = lenet_utils.inference(self.image_ph)
      self.labels = tf.nn.softmax(self.logits)

      if not is_training:
        predictions = tf.argmax(self.labels, axis=1)
        groundtruth = tf.argmax(self.hard_label_ph, axis=1)
        accuracy_list = tf.equal(predictions, groundtruth)
        self.accuracy = tf.reduce_mean(tf.cast(accuracy_list, tf.float32))
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(std_scope):
          continue
        print('%-64s added to STD saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)

      self.global_step = global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.Variable(flags.std_learning_rate, trainable=False)

      # pre train
      pre_losses = self.get_pre_losses()
      print('#pre_losses wo regularization=%d' % (len(pre_losses)))
      pre_losses.extend(self.get_regularization_losses())
      print('#pre_losses wt regularization=%d' % (len(pre_losses)))
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % std_scope)
      self.pre_train = lenet_utils.get_train_op(self.pre_loss, global_step)

      # kd train
      kd_losses = self.get_kd_losses(flags)
      print('#kd_losses wo regularization=%d' % (len(kd_losses)))
      kd_losses.extend(self.get_regularization_losses())
      print('#kd_losses wt regularization=%d' % (len(kd_losses)))
      self.kd_loss = tf.add_n(kd_losses, name='%s_kd_loss' % std_scope)
      # self.kd_train = lenet_utils.get_train_op(self.kd_loss, global_step)
      kd_optimizer = utils.get_opt(flags, self.learning_rate)
      self.kd_train = kd_optimizer.minimize(self.kd_loss, global_step=global_step)

      # gan train
      gan_losses = self.get_gan_losses()
      print('#gan_losses wo regularization=%d' % (len(gan_losses)))
      gan_losses.extend(self.get_regularization_losses())
      print('#gan_losses wt regularization=%d' % (len(gan_losses)))
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % std_scope)
      # self.gan_train = lenet_utils.get_train_op(self.gan_loss, global_step)
      gan_optimizer = utils.get_opt(flags, self.learning_rate)
      self.gan_train = gan_optimizer.minimize(self.gan_loss, global_step=global_step)

      # kdgan train
      kdgan_losses = self.get_kdgan_losses(flags)
      print('#kdgan_losses wo regularization=%d' % (len(kdgan_losses)))
      kdgan_losses.extend(self.get_regularization_losses())
      print('#kdgan_losses wt regularization=%d' % (len(kdgan_losses)))
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % std_scope)
      self.kdgan_train = lenet_utils.get_train_op(self.kdgan_loss, global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.std_scope):
        continue
      print('STD regularization=%s' % (regularization_loss.name))
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_hard_loss(self):
    hard_loss = lenet_utils.loss(self.logits, self.hard_label_ph)
    return hard_loss

  def get_pre_losses(self):
    pre_losses = [self.get_hard_loss()]
    return pre_losses

  def get_kd_losses(self, flags):
    hard_loss = self.get_hard_loss()
    hard_loss *= (1.0 - flags.kd_soft_pct)
    kd_losses = [hard_loss]
    
    if flags.kd_model == 'mimic':
      soft_loss = tf.nn.l2_loss(self.soft_logit_ph - self.logits)
      soft_loss *= (0.1 * flags.kd_soft_pct / flags.batch_size)
      kd_losses.append(soft_loss)
    elif flags.kd_model == 'distn':
      tch_logits = tf.scalar_mul(1.0 / flags.temperature, self.soft_logit_ph)
      std_logits = tf.scalar_mul(1.0 / flags.temperature, self.logits)
      soft_loss = tf.scalar_mul(0.5, tf.square(std_logits - tch_logits))
      soft_loss = tf.reduce_sum(soft_loss) * flags.kd_soft_pct / flags.batch_size
      kd_losses.append(soft_loss)
    elif flags.kd_model == 'noisy':
      noisy = np.float32(np.ones((flags.batch_size, flags.num_label)))
      noisy = tf.nn.dropout(noisy, keep_prob=(1.0 - flags.noisy_ratio))
      noisy += tf.random_normal((flags.batch_size, flags.num_label), stddev=flags.noisy_sigma)
      tch_logits = tf.multiply(self.soft_logit_ph, noisy)
      soft_loss = tf.nn.l2_loss(tch_logits - self.logits)
      soft_loss *= (0.1 * flags.kd_soft_pct / flags.batch_size)
      kd_losses.append(soft_loss)
    else:
      raise ValueError('bad kd model %s', flags.kd_model)
    # print('#kd_losses wo regularization=%d' % (len(kd_losses)))
    return kd_losses

  def get_gan_losses(self):
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    gan_losses = [tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)]
    return gan_losses

  def get_kdgan_losses(self, flags):
    kdgan_losses = []
    for gan_loss in self.get_gan_losses():
      gan_loss *= (1.0 - flags.intelltch_weight)
      kdgan_losses.append(gan_loss)
    for kd_loss in self.get_kd_losses(flags):
      kd_loss *= flags.distilled_weight
      kdgan_losses.append(kd_loss)
    return kdgan_losses
