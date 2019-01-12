from kdgan import config, utils
from kdgan.gen_model import GEN

import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

from PIL import Image

tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'l2 coefficient')
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

id_to_label = utils.load_id_to_label()

def check_tfrecord(bt_list, batch_size):
    id_to_label = utils.load_id_to_label()
    id_to_token = utils.load_id_to_token()

    user_bt, image_bt, text_bt, label_bt, image_file_bt = bt_list
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for t in range(3):
                user_np, image_np, text_np, label_np, image_file_np = sess.run(
                        [user_bt, image_bt, text_bt, label_bt, image_file_bt])
                # for b in range(batch_size):
                #     print('{0}\n{0}'.format('#'*80))
                #     print(user_np[b])
                #     num_token = text_np[b].shape[0]
                #     tokens = [id_to_token[text_np[b, i]] for i in range(num_token)]
                #     print(tokens)
                #     label_vt = label_np[b,:]
                #     label_ids = [i for i, l in enumerate(label_vt) if l != 0]
                #     labels = [id_to_label[label_id] for label_id in label_ids]
                #     print(labels)
                #     print(image_file_np[b])
                #     print('{0}\n{0}'.format('#'*80))
                #     input()
                print(user_np.shape, image_np.shape, text_np.shape, label_np.shape, image_file_np.shape)

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
    print('#label={}'.format(config.num_label))
    gen_t = GEN(flags, is_training=True)
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    gen_v = GEN(flags, is_training=False)

    ts_list_t = utils.decode_tfrecord(config.train_tfrecord, shuffle=True)
    ts_list_v = utils.decode_tfrecord(config.valid_tfrecord, shuffle=False)
    bt_list_t = utils.generate_batch(gen_t, ts_list_t, config.train_batch_size)
    bt_list_v = utils.generate_batch(gen_v, ts_list_v, config.valid_batch_size)
    # check_tfrecord(bt_list_t, config.train_batch_size)
    # check_tfrecord(bt_list_v, config.valid_batch_size)

    user_bt_t, image_bt_t, text_bt_t, label_bt_t, image_file_bt_t = bt_list_t
    user_bt_v, image_bt_v, text_bt_v, label_bt_v, image_file_bt_v = bt_list_v

    best_hit_v = -np.inf
    init_op = tf.global_variables_initializer()
    start = time.time()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
        sess.run(init_op)
        gen_t.init_fn(sess)
        with slim.queues.QueueRunners(sess):
            for batch_t in range(num_batch_t):
                image_np_t, label_np_t = sess.run([image_bt_t, label_bt_t])
                feed_dict = {gen_t.image_ph:image_np_t, gen_t.label_ph:label_np_t}
                _, summary = sess.run([gen_t.train_op, gen_t.summary_op], feed_dict=feed_dict)
                writer.add_summary(summary, batch_t)

                if (batch_t + 1) % int(config.train_data_size / config.train_batch_size) != 0:
                    continue

                hit_v = []
                image_file_v = set()
                for batch_v in range(num_batch_v):
                    image_np_v, label_np_v, image_file_np_v = sess.run([image_bt_v, label_bt_v, image_file_bt_v])
                    feed_dict = {gen_v.image_ph:image_np_v}
                    logit_np_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
                    for image_file in image_file_np_v:
                        image_file_v.add(image_file)
                    hit_bt = compute_hit(logit_np_v, label_np_v, flags.cutoff)
                    hit_v.append(hit_bt)
                hit_v = np.mean(hit_v)

                total_time = time.time() - start
                avg_batch = total_time / (batch_t + 1)
                avg_epoch = avg_batch * (config.train_data_size / config.train_batch_size)
                s = '{0} hit={1:.4f} tot={2:.0f}s avg={3:.0f}s'
                s = s.format(batch_t, hit_v, total_time, avg_epoch)
                print(s)

                if hit_v < best_hit_v:
                    continue
                best_hit_v = hit_v
                ckpt_file = path.join(config.ckpt_dir, 'gen_{}.ckpt'.format(flags.model_name))
                gen_t.saver.save(sess, ckpt_file)
    hit_file = path.join(config.temp_dir, '{}.hit'.format(flags.model_name))
    with open(hit_file, 'w') as fout:
        fout.write(best_hit_v)

if __name__ == '__main__':
    tf.app.run()