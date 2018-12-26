import cifar10_utils

import tensorflow as tf

class STD():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    image_shape = (flags.batch_size, flags.image_size, flags.image_size, flags.channels)
    self.image_ph = tf.placeholder(tf.float32, shape=image_shape)
    self.hard_label_ph = tf.placeholder(tf.int32, shape=(flags.batch_size))
    self.soft_logit_ph = tf.placeholder(tf.float32, shape=(flags.batch_size, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.std_scope = std_scope = 'std'
    with tf.variable_scope(std_scope) as scope:
      self.logits = cifar10_utils.inference(self.image_ph)
      self.labels = tf.nn.softmax(self.logits)

      if not is_training:
        top_k_op = tf.nn.in_top_k(self.logits, self.hard_label_ph, 1)
        self.accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(std_scope):
          continue
        print('%-50s added to STD saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)

      self.global_step = global_step = tf.Variable(0, trainable=False)
      
      # pre train
      pre_losses = self.get_pre_losses()
      print('#pre_losses wo regularization=%d' % (len(pre_losses)))
      pre_losses.extend(self.get_regularization_losses())
      print('#pre_losses wt regularization=%d' % (len(pre_losses)))
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % std_scope)
      self.pre_train = cifar10_utils.get_train_op(flags, self.pre_loss, global_step)

      # kd train
      kd_losses = self.get_kd_losses(flags)
      print('#kd_losses wo regularization=%d' % (len(kd_losses)))
      kd_losses.extend(self.get_regularization_losses())
      print('#kd_losses wt regularization=%d' % (len(kd_losses)))
      self.kd_loss = tf.add_n(kd_losses, name='%s_kd_loss' % std_scope)
      self.kd_train = cifar10_utils.get_train_op(flags, self.kd_loss, global_step)

      # gan train
      gan_losses = self.get_gan_losses()
      print('#gan_losses wo regularization=%d' % (len(gan_losses)))
      gan_losses.extend(self.get_regularization_losses())
      print('#gan_losses wt regularization=%d' % (len(gan_losses)))
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % std_scope)
      self.gan_train = cifar10_utils.get_train_op(flags, self.gan_loss, global_step)

      # kdgan train
      kdgan_losses = self.get_kdgan_losses(flags)
      print('#kdgan_losses wo regularization=%d' % (len(kdgan_losses)))
      kdgan_losses.extend(self.get_regularization_losses())
      print('#kdgan_losses wt regularization=%d' % (len(kdgan_losses)))
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % std_scope)
      self.kdgan_train = cifar10_utils.get_train_op(flags, self.kdgan_loss, global_step)

  def get_hard_loss(self):
    hard_loss = cifar10_utils.loss(self.logits, self.hard_label_ph)
    return hard_loss

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.std_scope):
        continue
      print('regularization_loss.name', regularization_loss.name)
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_pre_losses(self):
    pre_losses = [self.get_hard_loss()]
    return pre_losses

  def get_kd_losses(self, flags):
    hard_loss = self.get_hard_loss()
    hard_loss *= (1.0 - flags.kd_soft_pct)
    kd_losses = [hard_loss]
    if flags.kd_model == 'mimic':
      # soft_loss = tf.nn.l2_loss(self.soft_logit_ph - self.logits)
      soft_loss = tf.losses.mean_squared_error(self.soft_logit_ph, self.logits)
      soft_loss *= flags.kd_soft_pct
    elif flags.kd_model == 'distn':
      std_logits = self.logits * (1.0 / flags.temperature)
      tch_logits = self.soft_logit_ph * (1.0 / flags.temperature)

      # soft_loss = tf.losses.mean_squared_error(tch_logits, std_logits)
      # soft_loss *= pow(flags.temperature, 2.0)

      std_labels = tf.nn.softmax(std_logits)
      tch_labels = tf.nn.softmax(tch_logits)
      # soft_loss = -1.0 * tf.reduce_mean(tf.log(std_labels) * tch_labels)
      soft_loss = tf.losses.sigmoid_cross_entropy(tch_labels, tf.log(std_labels))

      soft_loss *= flags.kd_soft_pct
    else:
      raise ValueError('bad kd model %s', flags.kd_model)
    kd_losses.append(soft_loss)
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
