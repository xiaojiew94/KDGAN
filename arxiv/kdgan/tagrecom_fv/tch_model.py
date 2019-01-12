from kdgan import config
from kdgan import utils

from nets import nets_factory

import tensorflow as tf
from tensorflow.contrib import slim

class TCH():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training

    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size))
    self.text_ph = tf.placeholder(tf.int64, shape=(None, None))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_logit_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.tch_scope = tch_scope = 'tch'
    model_scope = nets_factory.arg_scopes_map[flags.image_model]
    vocab_size = utils.get_vocab_size(flags.dataset)
    with tf.variable_scope(tch_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.image_weight_decay)):
        iembed = self.image_ph
        iembed = slim.dropout(iembed, flags.image_keep_prob, is_training=is_training)

      with slim.arg_scope([slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(flags.text_weight_decay)):
        wembed = slim.variable('wembed',
            shape=[vocab_size, flags.embedding_size],
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        tembed = tf.nn.embedding_lookup(wembed, self.text_ph)
        tembed = tf.reduce_mean(tembed, axis=-2)

      with slim.arg_scope([slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(flags.tch_weight_decay),
          biases_initializer=tf.zeros_initializer()):
        # cembed = tf.concat([tembed], 1)
        cembed = tf.concat([iembed, tembed], 1)
        self.logits = slim.fully_connected(cembed, flags.num_label, activation_fn=None)

      self.labels = tf.nn.softmax(self.logits)

      if not is_training:
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(tch_scope):
          continue
        print('%-50s added to TCH saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)

      self.global_step = global_step = tf.Variable(0, trainable=False)
      tn_size = utils.get_tn_size(flags.dataset)
      learning_rate = flags.tch_learning_rate
      self.learning_rate = utils.get_lr(flags, tn_size, global_step, learning_rate, tch_scope)

      # pre train
      pre_losses = self.get_pre_losses()
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % tch_scope)
      pre_losses.extend(self.get_regularization_losses())
      print('#pre_losses wt regularization=%d' % (len(pre_losses)))
      pre_optimizer = utils.get_opt(flags, self.learning_rate)
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

      # kdgan train
      kdgan_losses = self.get_kdgan_losses(flags)
    # print('#kdgan_losses wo regularization=%d' % (len(kdgan_losses)))
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % tch_scope)
      kdgan_optimizer = utils.get_opt(flags, self.learning_rate)
      # self.kdgan_update = kdgan_optimizer.minimize(self.kdgan_loss, global_step=global_step)
      gvs = kdgan_optimizer.compute_gradients(self.kdgan_loss, var_list)
      cgvs = [(tf.clip_by_norm(gv[0], config.max_norm), gv[1]) for gv in gvs]
      self.kdgan_update = kdgan_optimizer.apply_gradients(cgvs, global_step=global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.tch_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_hard_loss(self):
    hard_loss = tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits)
    return hard_loss

  def get_pre_losses(self):
    pre_losses = [self.get_hard_loss()]
    # print('#pre_losses wo regularization=%d' % (len(pre_losses)))
    return pre_losses

  def get_kd_losses(self, flags):
    kd_losses = []
    if flags.kd_model == 'mimic':
      soft_loss = tf.nn.l2_loss(self.soft_logit_ph - self.logits) / flags.batch_size
      kd_losses.append(soft_loss)
    elif flags.kd_model == 'distn':
      hard_loss = self.get_hard_loss()
      hard_loss *= (1.0 - flags.kd_soft_pct) / flags.batch_size
      gen_logits = self.logits * (1.0 / flags.temperature)
      tch_logits = self.soft_logit_ph * (1.0 / flags.temperature)
      soft_loss = tf.losses.mean_squared_error(tch_logits, gen_logits)
      soft_loss *= (pow(flags.temperature, 2.0) * flags.kd_soft_pct) / flags.batch_size
      kd_losses.extend([hard_loss, soft_loss])
    elif flags.kd_model == 'noisy':
      # self.noisy = noisy = tf.multiply(noisy_mask, gaussian)
      # tch_logits = tf.multiply(self.soft_logit_ph, tf.tile(noisy, tf.constant([1, flags.num_label])))
      # soft_loss = tf.nn.l2_loss(tch_logits - self.logits) / flags.batch_size
      # kd_losses.append(soft_loss)
      noisy = np.float32(np.ones((flags.batch_size, flags.num_label)))
      noisy = tf.nn.dropout(noisy, keep_prob=(1.0 - flags.noisy_ratio))
      noisy += tf.random_normal((flags.batch_size, flags.num_label), stddev=flags.noisy_sigma)
      tch_logits = tf.multiply(self.soft_logit_ph, noisy)
      soft_loss = tf.nn.l2_loss(tch_logits - self.logits) / flags.batch_size
      kd_losses.append(soft_loss)
    else:
      raise ValueError('bad kd model %s', flags.kd_model)
    return kd_losses

  def get_kdgan_losses(self, flags):
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    gan_loss = tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)
    gan_loss *= flags.intelltch_weight
    kdgan_losses = [gan_loss]
    for kd_loss in self.get_kd_losses(flags):
      kd_loss *= flags.intellstd_weight
      kdgan_losses.append(kd_loss)
    return kdgan_losses

