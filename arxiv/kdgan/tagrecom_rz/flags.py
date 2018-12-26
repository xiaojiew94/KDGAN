import tensorflow as tf

# evaluation
tf.app.flags.DEFINE_integer('cutoff', 1, '')
# dataset
tf.app.flags.DEFINE_string('dataset', None, '')
tf.app.flags.DEFINE_integer('num_label', 100, '')
# model
tf.app.flags.DEFINE_string('image_model', None, '')
tf.app.flags.DEFINE_float('image_keep_prob', 5e-1, '')
tf.app.flags.DEFINE_float('image_weight_decay', 1e-4, 'l2 coefficient')
tf.app.flags.DEFINE_float('text_weight_decay', 0e-4, 'l2 coefficient')
tf.app.flags.DEFINE_float('tch_weight_decay', 0e-4, 'l2 coefficient')
tf.app.flags.DEFINE_string('dis_model_ckpt', None, '')
tf.app.flags.DEFINE_string('gen_model_ckpt', None, '')
tf.app.flags.DEFINE_string('tch_model_ckpt', None, '')
tf.app.flags.DEFINE_integer('feature_size', 4096, '')
tf.app.flags.DEFINE_integer('embedding_size', 10, '')
tf.app.flags.DEFINE_string('epk_learning_curve_p', None, '')
# optimization
tf.app.flags.DEFINE_integer('num_negative', 1, '')
tf.app.flags.DEFINE_integer('num_positive', 1, '')
tf.app.flags.DEFINE_float('epk_train', 1.0, '')
tf.app.flags.DEFINE_float('epk_valid', 0.0, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_epoch', 20, '')
tf.app.flags.DEFINE_integer('num_dis_epoch', 10, '')
tf.app.flags.DEFINE_integer('num_gen_epoch', 5, '')
tf.app.flags.DEFINE_integer('num_tch_epoch', 5, '')
tf.app.flags.DEFINE_float('dis_learning_rate', 1e-2, '')
tf.app.flags.DEFINE_float('gen_learning_rate', 1e-2, '')
tf.app.flags.DEFINE_float('tch_learning_rate', 1e-2, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, '')
tf.app.flags.DEFINE_float('end_learning_rate', 1e-5, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 1e1, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exp', 'exp|fix|ply')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'adam|rmsprop|sgd')
# kdgan
tf.app.flags.DEFINE_float('kd_soft_pct', 0.3, '')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_string('kd_model', 'mimic', 'mimic|distn')
tf.app.flags.DEFINE_float('alpha', 0.5, '')
tf.app.flags.DEFINE_float('beta', 0.5, '')
tf.app.flags.DEFINE_float('gamma', 0.0, '')

flags = tf.app.flags.FLAGS
