from kdgan import config

from nets import nets_factory

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class GEN():
    def __init__(self, flags, is_training=True):
        self.is_training = is_training
        self.preprocessing_name = (flags.preprocessing_name or flags.model_name)

        network_fn = nets_factory.get_network_fn(flags.model_name,
                num_classes=config.num_label,
                weight_decay=flags.weight_decay,
                is_training=is_training)
        self.image_size = network_fn.default_image_size

        self.image_ph = tf.placeholder(tf.float32,
                shape=(None, self.image_size, self.image_size, config.channels))
        self.label_ph = tf.placeholder(tf.float32,
                shape=(None, config.num_label))
        
        self.logits, end_points = network_fn(self.image_ph)

        if not is_training:
            return

        # global_step = tf.train.create_global_step()
        global_step = tf.train.get_global_step()
        decay_steps = int(config.train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
                global_step, decay_steps, flags.learning_rate_decay_factor,
                staircase=True, name='exponential_decay_learning_rate')

        tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits)
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        losses.extend(regularization_losses)
        loss = tf.add_n(losses, name='loss')
        total_loss = tf.losses.get_total_loss(name='total_loss')
        diff = tf.subtract(loss, total_loss)


        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('diff', diff)
        self.summary_op = tf.summary.merge_all()

        exclusions = [scope.strip() for scope in flags.checkpoint_exclude_scopes.split(',')]

        variables_to_restore = []
        for variable in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if variable.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(variable)
            else:
                num_params = 1
                for dim in variable.shape:
                    num_params *= dim.value
                print('randinit {}\t({} params)'.format(variable.name, num_params))

        scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        self.save_dict = {}
        for variable in variables_to_train:
            if not variable.name.startswith('vgg_16'):
                continue
            num_params = 1
            for dim in variable.shape:
                num_params *= dim.value
            print('trainable {}\t({} params)'.format(variable.name, num_params))
            self.save_dict[variable.name] = variable
        self.saver = tf.train.Saver(self.save_dict)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_op = optimizer.minimize(loss, var_list=variables_to_train, global_step=global_step)
        
        self.init_fn = slim.assign_from_checkpoint_fn(flags.checkpoint_path, variables_to_restore)






