from kdgan import config

from nets import nets_factory
from nets import vgg

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class GEN():
    def __init__(self, flags, is_training=True):
        self.is_training = is_training

        self.image_ph = tf.placeholder(tf.float32,
                shape=(None, flags.feature_size))
        self.label_ph = tf.placeholder(tf.float32,
                shape=(None, config.num_label))

        dropout_keep_prob = 0.5
        net = self.image_ph
        net = tf.expand_dims(tf.expand_dims(net, axis=1), axis=1)
        with slim.arg_scope(vgg.vgg_arg_scope(
                weight_decay=flags.weight_decay)):
            net = slim.dropout(net, dropout_keep_prob, 
                    is_training=is_training,
                    scope='dropout7')
            net = slim.conv2d(net, config.num_label, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='generator')
            self.logits = tf.squeeze(net)

        if not is_training:
            return

        train_data_size = utils.get_tn_size(flags.dataset)
        global_step = tf.train.get_global_step()
        decay_steps = int(train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
        self.learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
            global_step, decay_steps, flags.learning_rate_decay_factor,
            staircase=True, name='exponential_decay_learning_rate')

        tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits)
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        # for loss in losses:
        #     print(loss)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # for regularization_loss in regularization_losses:
        #     print(regularization_loss)
        losses.extend(regularization_losses)
        loss = tf.add_n(losses, name='loss')
        total_loss = tf.losses.get_total_loss(name='total_loss')
        diff = tf.subtract(loss, total_loss)


        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('diff', diff)
        self.summary_op = tf.summary.merge_all()

        variables_to_train = tf.trainable_variables()
        for variable in variables_to_train:
            num_params = 1
            for dim in variable.shape:
                num_params *= dim.value
            print('trainable {}\t({} params)'.format(variable.name, num_params))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_op = optimizer.minimize(loss, 
                global_step=global_step)





