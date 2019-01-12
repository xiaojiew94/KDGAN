from kdgan import config
from kdgan import metric
from kdgan import utils

import random

class YFCCDATA(object):
  def __init__(self, flags):
    tn_user_bt, tn_image_bt, tn_text_bt, tn_label_bt, _ = self.get_batch(flags, True)
    self.tn_image_bt, self.tn_text_bt, self.tn_label_bt = tn_image_bt, tn_text_bt, tn_label_bt

    vd_user_bt, vd_image_bt, vd_text_bt, vd_label_bt, _ = self.get_batch(flags, False)
    self.vd_image_bt, self.vd_text_bt, self.vd_label_bt = vd_image_bt, vd_text_bt, vd_label_bt

  def get_batch(self, flags, is_training):
    if is_training:
      single = False
      stage = 'train'
      shuffle = True
    else:
      single = True
      stage = 'valid'
      shuffle = False
    data_sources = utils.get_data_sources(flags, is_training=is_training, single=single)
    print('#tfrecord=%d for %s' % (len(data_sources), stage))
    ts_list = utils.decode_tfrecord(flags, data_sources, shuffle=shuffle)
    bt_list = utils.generate_batch(ts_list, flags.batch_size)
    return bt_list

  def next_batch(self, flags, sess):
    rand = random.random()
    tn_vd_border = 1.0 * flags.epk_train / (flags.epk_train + flags.epk_valid)
    if rand < tn_vd_border:
      image_bt, text_bt, label_bt = self.tn_image_bt, self.tn_text_bt, self.tn_label_bt
    else:
      image_bt, text_bt, label_bt = self.vd_image_bt, self.vd_text_bt, self.vd_label_bt
    image_np, text_np, label_np = sess.run([image_bt, text_bt, label_bt])
    return image_np, text_np, label_np

class YFCCEVAL(object):
  def __init__(self, flags):
    self.vd_image_np, self.vd_text_np, self.vd_label_np, _ = utils.get_valid_data(flags)

  def compute_prec(self, flags, sess, vd_model):
    if hasattr(vd_model, 'image_ph') and hasattr(vd_model, 'text_ph'):
      feed_dict = {
        vd_model.image_ph:self.vd_image_np,
        vd_model.text_ph:self.vd_text_np,
      }
    elif hasattr(vd_model, 'image_ph'):
      feed_dict = {vd_model.image_ph:self.vd_image_np}
    elif hasattr(vd_model, 'text_ph'):
      feed_dict = {vd_model.text_ph:self.vd_text_np}
    else:
      feed_dict = {}
    vd_logit_np = sess.run(vd_model.logits, feed_dict=feed_dict)
    prec = metric.compute_prec(vd_logit_np, self.vd_label_np, flags.cutoff)
    return prec












