from kdgan import config, utils

from nets import nets_factory
from nets import vgg

from tensorflow.contrib import slim
import numpy as np
import tensorflow as tf

class GEN():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_logit_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.gen_scope = gen_scope = 'gen'
    model_scope = nets_factory.arg_scopes_map[flags.image_model]
    with tf.variable_scope(gen_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.image_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.image_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, flags.num_label, activation_fn=None)
        self.logits = net

      self.labels = tf.nn.softmax(self.logits)

      if not is_training:
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(gen_scope):
          continue
        print('%-50s added to GEN saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)

      self.global_step = global_step = tf.Variable(0, trainable=False)
      tn_size = utils.get_tn_size(flags.dataset)
      learning_rate = flags.gen_learning_rate
      self.learning_rate = utils.get_lr(flags, tn_size, global_step, learning_rate, gen_scope)

      # pre train
      pre_losses = self.get_pre_losses()
      print('#pre_losses wo regularization=%d' % (len(pre_losses)))
      pre_losses.extend(self.get_regularization_losses())
      print('#pre_losses wt regularization=%d' % (len(pre_losses)))
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % gen_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate)
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

      # kd train
      kd_losses = self.get_kd_losses(flags)
      print('#kd_losses wo regularization=%d' % (len(kd_losses)))
      self.kd_loss = tf.add_n(kd_losses, name='%s_kd_loss' % gen_scope)
      kd_optimizer = utils.get_opt(flags, self.learning_rate)
      self.kd_update = kd_optimizer.minimize(self.kd_loss, global_step=global_step)

      # gan train
      gan_losses = self.get_gan_losses()
      print('#gan_losses wo regularization=%d' % (len(gan_losses)))
      gan_losses.extend(self.get_regularization_losses())
      print('#gan_losses wt regularization=%d' % (len(gan_losses)))
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % gen_scope)
      gan_optimizer = utils.get_opt(flags, self.learning_rate)
      self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=global_step)

      # kdgan train
      kdgan_losses = self.get_kdgan_losses(flags)
      print('#kdgan_losses wo regularization=%d' % (len(kdgan_losses)))
      kdgan_losses.extend(self.get_regularization_losses())
      print('#kdgan_losses wt regularization=%d' % (len(kdgan_losses)))
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % gen_scope)
      kdgan_optimizer = utils.get_opt(flags, self.learning_rate)
      # self.kdgan_update = kdgan_optimizer.minimize(self.kdgan_loss, global_step=global_step)
      gvs = kdgan_optimizer.compute_gradients(self.kdgan_loss, var_list)
      cgvs = [(tf.clip_by_norm(gv[0], config.max_norm), gv[1]) for gv in gvs]
      self.kdgan_update = kdgan_optimizer.apply_gradients(cgvs, global_step=global_step)

  def get_hard_loss(self):
    hard_loss = tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits)
    return hard_loss

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.gen_scope):
        continue
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
      gen_logits = self.logits * (1.0 / flags.temperature)
      tch_logits = self.soft_logit_ph * (1.0 / flags.temperature)

      # soft_loss = tf.losses.mean_squared_error(tch_logits, gen_logits)
      # soft_loss *= pow(flags.temperature, 2.0)

      gen_labels = tf.nn.softmax(gen_logits)
      tch_labels = tf.nn.softmax(tch_logits)
      # soft_loss = -1.0 * tf.reduce_mean(tf.log(gen_labels) * tch_labels)
      soft_loss = tf.losses.sigmoid_cross_entropy(tch_labels, tf.log(gen_labels))

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
      gan_loss *= flags.beta
      kdgan_losses.append(gan_loss)
    for kd_loss in self.get_kd_losses(flags):
      kd_loss *= (1.0 - flags.beta)
      kdgan_losses.append(kd_loss)
    return kdgan_losses
