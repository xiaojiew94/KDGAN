from os import path

from collections import Counter
from tensorflow.contrib import slim

import numpy as np
import tensorflow as tf

tf.flags.DEFINE_boolean('dev', False, 'dev')
tf.flags.DEFINE_string('facebook_infile', None, '')
tf.flags.DEFINE_string('ngrams', None, '--ngrams=2,3,4,5')

tf.flags.DEFINE_string('train_tfrecord', None, '')
tf.flags.DEFINE_string('valid_tfrecord', None, '')
tf.flags.DEFINE_string('label_file', None, '')
tf.flags.DEFINE_string('vocab_file', None, '')
tf.flags.DEFINE_string('logs_dir', None, '')

tf.flags.DEFINE_integer('batch_size', 100, '')
tf.flags.DEFINE_integer('train_steps', 50000, '')
tf.flags.DEFINE_integer('valid_steps', 100, '')
tf.flags.DEFINE_integer('num_epochs', 1000, '')
tf.flags.DEFINE_float('learning_rate', 0.01, '')

tf.flags.DEFINE_integer('num_oov_vocab_buckets', 20,
        'number of hash buckets to use for OOV words')
tf.flags.DEFINE_integer('embedding_dimension', 10,
        'dimension of word embedding')
tf.flags.DEFINE_boolean('use_ngrams', False,
        'use character ngrams in embedding')
tf.flags.DEFINE_integer('num_ngram_buckets', 1000000,
        'number of hash buckets for ngrams')
tf.flags.DEFINE_integer('ngram_embedding_dimension', 10,
        'dimension of word embedding')

tf.flags.DEFINE_integer('num_threads', 1,
        'number of reader threads')

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)
if FLAGS.dev:
    tf.logging.set_verbosity(tf.logging.DEBUG)

TEXT_KEY = 'text'
LABELS_KEY = 'labels'
NGRAMS_KEY = 'ngrams'
# DEFAULT_WORD = ' '
DEFAULT_WORD = len(open(FLAGS.vocab_file).readlines())

def parse_ngrams(ngrams):
    ngrams = [int(g) for g in ngrams.split(',')]
    ngrams = [g for g in ngrams if (g > 1 and g < 7)]
    return ngrams

def generate_ngrams(words, ngrams):
    nglist = []
    for ng in ngrams:
        for word in words:
            nglist.extend([word[n:n+ng] for n in range(len(word)-ng+1)])
    return nglist

def load_label_to_id():
    fin = open(FLAGS.label_file)
    labels = [label.strip() for label in fin.readlines()]
    fin.close()
    label_to_id = dict(zip(labels, range(len(labels))))
    return label_to_id

def load_id_to_label():
    label_to_id = load_label_to_id()
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    return id_to_label

def load_sth_to_id(infile):
    with open(infile) as fin:
        sth_list = [sth.strip() for sth in fin.readlines()]
    sth_to_id = dict(zip(sth_list, range(len(sth_list))))
    return sth_to_id

def load_vocab_to_id():
    vocab_to_id = load_sth_to_id(FLAGS.vocab_file)
    return vocab_to_id

def parse_facebook_infile(infile, ngrams):
    label_prefix = '__label__'
    examples = []
    for line in open(infile):
        words = line.split()
        labels = []
        for word in words:
            if word.startswith(label_prefix):
                label = word[len(label_prefix):]
                labels.append(label)
            else:
                break
        text = words[len(labels):]
        if len(labels) == 0:
            print('no labels')
            exit()
        if len(text) == 0:
            print('no text')
            exit()
        example = {LABELS_KEY: labels, TEXT_KEY:text}
        if ngrams:
            example[NGRAMS_KEY] = generate_ngrams(text, ngrams)
        examples.append(example)
    return examples

def build_tfrecord(example, label_to_id, vocab_to_id):
    text = example[TEXT_KEY]
    labels = example[LABELS_KEY]
    ngrams = example.get(NGRAMS_KEY, None)
    record = tf.train.Example()
    
    # text = [tf.compat.as_bytes(x) for x in text]
    # record.features.feature[TEXT_KEY].bytes_list.value.extend(text)
    unk = len(vocab_to_id)
    text = [vocab_to_id.get(x, unk) for x in text]
    record.features.feature[TEXT_KEY].int64_list.value.extend(text)
    
    # labels = [tf.compat.as_bytes(x) for x in labels]
    # record.features.feature[LABELS_KEY].bytes_list.value.extend(labels)
    label_ids = [label_to_id[label] for label in labels]
    labels = np.zeros((len(label_to_id),), dtype=np.float32)
    labels[label_ids] = 1
    record.features.feature[LABELS_KEY].float_list.value.extend(labels)

    if ngrams is not None:
        ngrams = [tf.compat.as_bytes(x) for x in ngrams]
        record.features.feature[NGRAMS_KEY].bytes_list.value.extend(ngrams)
    return record

def write_examples(examples, tfrecord_file):
    label_to_id = load_label_to_id()
    vocab_to_id = load_vocab_to_id()
    # print(vocab_to_id)
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for n, example in enumerate(examples):
        record = build_tfrecord(example, label_to_id, vocab_to_id)
        writer.write(record.SerializeToString())

def write_vocab(examples, vocab_file, label_file):
    words = Counter()
    labels = set()
    for example in examples:
        words.update(example[TEXT_KEY])
        labels.update(example[LABELS_KEY])
    with open(vocab_file, 'w') as fout:
        for word in words.most_common():
            fout.write(word[0] + '\n')
    with open(label_file, 'w') as fout:
        labels = sorted(list(labels))
        for label in labels:
            fout.write(str(label) + '\n')

def cleanse():
    if not FLAGS.facebook_infile:
        print('no --facebook_infile')
        exit()
    if not FLAGS.label_file:
        print('no --label_file')
        exit()
    if not FLAGS.vocab_file:
        print('no --vocab_file')
        exit()
    ngrams = None
    if FLAGS.ngrams:
        ngrams = parse_ngrams(FLAGS.ngrams)

    # vocab_file = path.join(FLAGS.facebook_infile + '.vocab')
    # write_vocab(vocab_file)

    examples = parse_facebook_infile(FLAGS.facebook_infile, ngrams)
    tfrecord_file = path.join(FLAGS.facebook_infile + '.tfrecord')
    write_examples(examples, tfrecord_file)


def get_parse_spec(use_ngrams, num_label):
    parse_spec = {
        # TEXT_KEY:tf.VarLenFeature(dtype=tf.string),
        TEXT_KEY:tf.VarLenFeature(dtype=tf.int64),
        LABELS_KEY:tf.FixedLenFeature([num_label], tf.float32, default_value=tf.zeros([num_label], dtype=tf.float32)),
    }
    if use_ngrams:
        parse_spec[NGRAMS_KEY] = tf.VarLenFeature(dtype=tf.string)
    return parse_spec

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

def precision(logits, labels, cutoff):
    def normalize(cutoff, num_label):
        return min(cutoff, num_label)
    prec = evaluate(logits, labels, cutoff, normalize)
    # print('prec={0:.4f}'.format(prec))
    return prec

def recall(logits, labels, cutoff):
    def normalize(cutoff, num_label):
        return num_label
    rec = evaluate(logits, labels, cutoff, normalize)
    # print('rec={0:.4f}'.format(rec))
    return rec

def train():
    vocab_size = len(open(FLAGS.vocab_file).readlines())
    id_to_label = load_id_to_label()
    num_label = len(id_to_label)
    print('#vocab={} #label={}'.format(vocab_size, num_label))

    parse_spec = get_parse_spec(FLAGS.use_ngrams, num_label)
    features = tf.contrib.learn.read_batch_features(
            FLAGS.train_tfrecord,
            FLAGS.batch_size,
            parse_spec,
            tf.TFRecordReader,
            num_epochs=FLAGS.num_epochs,
            reader_num_threads=FLAGS.num_threads)
    text_ts = tf.sparse_tensor_to_dense(features[TEXT_KEY], default_value=DEFAULT_WORD)
    label_ts = features.pop(LABELS_KEY)
    
    # text_ph = tf.placeholder(tf.string, shape=(None, None))
    text_ph = tf.placeholder(tf.int64, shape=(None, None))
    label_ph = tf.placeholder(tf.float32, shape=(None, num_label))
    # text_lookup_table = tf.contrib.lookup.index_table_from_file(
    #         FLAGS.vocab_file, FLAGS.num_oov_vocab_buckets, vocab_size)
    # text_ids = text_lookup_table.lookup(text_ph)
    text_ids = text_ph
    # text_embedding_w = tf.Variable(tf.random_uniform([vocab_size + FLAGS.num_oov_vocab_buckets, FLAGS.embedding_dimension], -0.1, 0.1))
    text_embedding_w = tf.Variable(tf.random_uniform([vocab_size + 1, FLAGS.embedding_dimension], -0.1, 0.1))
    text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(text_embedding_w, text_ids), axis=-2)
    input_layer = text_embedding
    logits_ts = tf.contrib.layers.fully_connected(inputs=input_layer, num_outputs=num_label, activation_fn=None)
    loss_ts = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ph, logits=logits_ts))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optimizer.minimize(loss_ts, global_step=tf.train.get_global_step())
    var_init = tf.global_variables_initializer()
    tab_init = tf.tables_initializer()

    tf.summary.scalar('loss', loss_ts)
    summary_op = tf.summary.merge_all()


    features_v = tf.contrib.learn.read_batch_features(
            FLAGS.valid_tfrecord,
            FLAGS.batch_size,
            parse_spec,
            tf.TFRecordReader,
            num_epochs=1,
            reader_num_threads=FLAGS.num_threads)
    text_ts_v = tf.sparse_tensor_to_dense(features_v[TEXT_KEY], default_value=DEFAULT_WORD)
    label_ts_v = features_v.pop(LABELS_KEY)
    
    from tensorflow.python.framework import errors
    from tensorflow.python.ops import variables
    from tensorflow.python.training import coordinator
    from tensorflow.python.training import queue_runner_impl
    with tf.Session() as sess:
      writer = tf.summary.FileWriter(FLAGS.logs_dir, graph=tf.get_default_graph())

      sess.run(variables.local_variables_initializer())
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)
      sess.run(var_init)
      sess.run(tab_init)
      total_size = 0
      try:
        while not coord.should_stop():
            # feature_np, label_np = sess.run([features, label_ts])
            # text_np = feature_np[TEXT_KEY]
            # print(type(text_np), text_np.shape, type(label_np), label_np.shape)
            # for i in range(FLAGS.batch_size):
            #     label_ids = [j for j in range(num_label) if label_np[i,j] != 0]
            #     labels = [id_to_label[label_id] for label_id in label_ids]
            #     text = [text_np[i,j].decode('utf-8') for j in range(text_np.shape[1]) if text_np[i,j] != b' ']
            #     text = ' '.join(text)
            #     print(str(text), labels)
            #     input()
            # input()
            for train_step in range(1000000):
                text_np, label_np = sess.run([text_ts, label_ts])
                total_size += FLAGS.batch_size
                # print(type(text_np), text_np.shape, type(label_np), label_np.shape)
                # for i in range(FLAGS.batch_size):
                #     label_ids = [j for j in range(num_label) if label_np[i,j] != 0]
                #     labels = [id_to_label[label_id] for label_id in label_ids]
                #     text = [text_np[i,j].decode('utf-8') for j in range(text_np.shape[1]) if text_np[i,j] != b' ']
                #     text = ' '.join(text)
                #     print(str(text), labels)
                #     input()
                
                feed_dict = {text_ph:text_np, label_ph:label_np}
                _, loss, summary = sess.run([train_op, loss_ts, summary_op], feed_dict=feed_dict)
                if (train_step + 1) % 100 == 0:
                    writer.add_summary(summary, train_step)
                    print('#{0} loss={1:.4f}'.format(train_step, loss))
      except errors.OutOfRangeError:
        print('total={}'.format(total_size))
        cutoff = 3
        prec_v, rec_v = [], []
        for valid_step in range(int(2000 / FLAGS.batch_size)):
            text_np, label_np = sess.run([text_ts_v, label_ts_v])
            feed_dict = {text_ph:text_np, label_ph:label_np}
            logits, = sess.run([logits_ts], feed_dict=feed_dict)
            prec_bt = precision(logits, label_np, cutoff)
            prec_v.append(prec_bt)
            rec_bt = recall(logits, label_np, cutoff)
            rec_v.append(rec_bt)
        prec_v, rec_v = np.mean(prec_v), np.mean(rec_v)
        print('prec={0:.4f} rec={1:.4f}'.format(prec_v, rec_v))
      finally:
        coord.request_stop()

      coord.join(threads)

def test():
    vocab_size = len(open(FLAGS.vocab_file).readlines())
    id_to_label = load_id_to_label()
    num_label = len(id_to_label)
    print('#vocab={} #label={}'.format(vocab_size, num_label))

    data_sources = [FLAGS.train_tfrecord,]
    is_training = True
    reader = tf.TFRecordReader
    keys_to_features = {
        TEXT_KEY:tf.VarLenFeature(dtype=tf.string),
        LABELS_KEY:tf.FixedLenFeature([num_label], tf.float32, default_value=tf.zeros([num_label], dtype=tf.float32)),
    }

    items_to_handlers = {
        'text':slim.tfexample_decoder.Tensor(TEXT_KEY, default_value=DEFAULT_WORD),
        'labels':slim.tfexample_decoder.Tensor(LABELS_KEY),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)
    num_samples = 1 # np.inf
    items_to_descriptions = {
        'text': 'text',
        'labels': 'labels',
    }
    dataset = slim.dataset.Dataset(
        data_sources=data_sources,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
    )
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=is_training)
    text_ts, labels_ts, = provider.get(['text', 'labels'])

    # with tf.Session() as sess:
    #     with slim.queues.QueueRunners(sess):
    #         for i in range(10000):
    #             text_np, labels_np = sess.run([text_ts, labels_ts])
    #             label_ids = [i for i in range(num_label) if labels_np[i] != 0]
    #             labels = [id_to_label[label_id] for label_id in label_ids]
    #             text = [text_np[i].decode('utf-8') for i in range(text_np.shape[0]) if text_np[i] != b' ']
    #             text = ' '.join(text)
    #             print(str(text), labels)
    #             input()

    text_bt, labels_bt = tf.train.batch(
            [text_ts, labels_ts], 
            batch_size=FLAGS.batch_size,
            dynamic_pad=True)

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for i in range(10000):
                text_np, labels_np = sess.run([text_bt, labels_bt])
                print(type(text_np), type(labels_np))
                print(text_np.shape, labels_np.shape)
                input()

def main(_):
    # cleanse()
    train()
    # test()

if __name__ == '__main__':
    tf.app.run()

