import tensorflow as tf
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('gen_model_p', None, '')
tf.app.flags.DEFINE_string('tch_model_p', None, '')
tf.app.flags.DEFINE_string('gan_model_p', None, '')
tf.app.flags.DEFINE_string('kdgan_model_p', None, '')
tf.app.flags.DEFINE_string('epsfile', None, '')
flags = tf.app.flags.FLAGS