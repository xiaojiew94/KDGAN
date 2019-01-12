from kdgan import config

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class TCH():
    def __init__(self, flags, is_training=True):
        self.is_training = is_training

        self.text_ph = tf.placeholder(tf.int64,
                shape=(None, None))
        self.label_ph = tf.placeholder(tf.float32,
                shape=(None, config.num_label))

        # text_embedding_w = tf.Variable(tf.random_uniform([config.vocab_size, flags.embedding_size], -0.1, 0.1))
        text_embedding_w = tf.get_variable('text_embedding_w',
                initializer=tf.random_uniform([config.vocab_size, flags.embedding_size], -0.1, 0.1))
        text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(text_embedding_w, self.text_ph), axis=-2)
        input_layer = text_embedding
        # self.logits = tf.contrib.layers.fully_connected(
        #         inputs=input_layer,
        #         num_outputs=config.num_label,
        #         activation_fn=None)
        self.logits = slim.fully_connected(input_layer, 
                config.num_label,
                activation_fn=None,
                scope='teacher')

        if not is_training:
            return

        self.save_dict = {}
        variables_to_train = tf.trainable_variables()
        for variable in variables_to_train:
            if variable.name.startswith('vgg_16'):
                continue
            num_params = 1
            for dim in variable.shape:
                num_params *= dim.value
            print('trainable {}\t({} params)'.format(variable.name, num_params))
            self.save_dict[variable.name] = variable
        self.saver = tf.train.Saver(self.save_dict)

        loss_ts = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.label_ph, logits=self.logits))
        global_step = tf.train.get_global_step()
        decay_steps = int(config.train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
                global_step, decay_steps, flags.learning_rate_decay_factor,
                staircase=True, name='exponential_decay_learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate)

        self.train_op = optimizer.minimize(loss_ts, 
                global_step=tf.train.get_global_step())

        tf.summary.scalar('loss', loss_ts)
        self.summary_op = tf.summary.merge_all()

