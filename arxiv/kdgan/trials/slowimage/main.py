from kdgan import config, utils
from kdgan.gen_model import GEN
from kdgan.tch_model import TCH

import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

from PIL import Image

tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_integer('embedding_size', 10, '')
tf.app.flags.DEFINE_integer('num_epoch', 100, '')
tf.app.flags.DEFINE_float('init_learning_rate', 0.1, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0, '')
tf.app.flags.DEFINE_string('checkpoint_path', None, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,'')
tf.app.flags.DEFINE_string('trainable_scopes', None, '')
tf.app.flags.DEFINE_integer('cutoff', 3, '')
flags = tf.app.flags.FLAGS

num_batch_t = int(flags.num_epoch * config.train_data_size / config.train_batch_size)
num_batch_v = int(config.valid_data_size / config.valid_batch_size)
print('train#batch={} valid#batch={}'.format(num_batch_t, num_batch_v))

def evaluate(logits, labels, cutoff, normalize):
    predictions = np.argsort(-logits, axis=1)[:,:cutoff]
    batch_size, _ = labels.shape
    scores = []
    for batch in range(batch_size):
        label_bt = labels[batch,:]
        label_bt = np.nonzero(label_bt)[0]
        prediction_bt = predictions[batch,:]
        num_label = len(label_bt)
        present = 0
        for label in label_bt:
            if label in prediction_bt:
                present += 1
        score = present
        if score > 0:
            score *= (1.0 / normalize(cutoff, num_label))
        # print('score={0:.4f}'.format(score))
        scores.append(score)
    score = np.mean(scores)
    return score

def compute_hit(logits, labels, cutoff):
    def normalize(cutoff, num_label):
        return min(cutoff, num_label)
    hit = evaluate(logits, labels, cutoff, normalize)
    # print('hit={0:.4f}'.format(hit))
    return hit

def main(_):
    global_step = tf.train.create_global_step()
    print('#label={}'.format(config.num_label))
    gen_t = GEN(flags, is_training=True)
    tch_t = TCH(flags, is_training=True)
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    gen_v = GEN(flags, is_training=False)
    tch_v = TCH(flags, is_training=False)

    ts_list_t = utils.decode_tfrecord(config.train_tfrecord, shuffle=True)
    ts_list_v = utils.decode_tfrecord(config.valid_tfrecord, shuffle=False)
    bt_list_t = utils.generate_batch(gen_t, ts_list_t, config.train_batch_size)
    bt_list_v = utils.generate_batch(gen_v, ts_list_v, config.valid_batch_size)
    # check_tfrecord(bt_list_t, config.train_batch_size)
    # check_tfrecord(bt_list_v, config.valid_batch_size)

    user_bt_t, image_bt_t, text_bt_t, label_bt_t, file_bt_t = bt_list_t
    user_bt_v, image_bt_v, text_bt_v, label_bt_v, file_bt_v = bt_list_v

    best_hit_v = -np.inf
    init_op = tf.global_variables_initializer()
    start = time.time()
    gen_model_ckpt = path.join(config.ckpt_dir, 'gen_{}.ckpt'.format(flags.model_name))
    tch_model_ckpt = path.join(config.ckpt_dir, 'tch.ckpt'.format(flags.model_name))
    with tf.Session() as sess:
        # sess.run(init_op)
        gen_t.init_fn(sess)
        gen_t.saver.restore(sess, gen_model_ckpt)
        tch_t.saver.restore(sess, tch_model_ckpt)
        with slim.queues.QueueRunners(sess):
            image_hit_v, text_hit_v = [], []
            file_v = set()
            for batch_v in range(num_batch_v):
                image_np_v, text_np_v, label_np_v, file_np_v = sess.run(
                        [image_bt_v, text_bt_v, label_bt_v, file_bt_v])

                for file in file_np_v:
                    file_v.add(file)

                feed_dict = {gen_v.image_ph:image_np_v}
                logit_np_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
                image_hit_bt = compute_hit(logit_np_v, label_np_v, flags.cutoff)
                image_hit_v.append(image_hit_bt)

                feed_dict = {tch_v.text_ph:text_np_v}
                logit_np_v, = sess.run([tch_v.logits], feed_dict=feed_dict)
                text_hit_bt = compute_hit(logit_np_v, label_np_v, flags.cutoff)
                text_hit_v.append(text_hit_bt)
                
            image_hit_v = np.mean(image_hit_v)
            text_hit_v = np.mean(text_hit_v)

            print('#file={}'.format(len(file_v)))

            total_time = time.time() - start
            print(image_hit_v)
            print(text_hit_v)
            print(total_time)

if __name__ == '__main__':
    tf.app.run()