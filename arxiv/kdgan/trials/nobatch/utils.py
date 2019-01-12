from kdgan import config, metric

import os

import numpy as np
import tensorflow as tf

from os import path
from preprocessing import preprocessing_factory
from tensorflow.contrib import slim


def create_if_nonexist(outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)

def save_collection(coll, outfile):
    with open(outfile, 'w') as fout:
        for elem in coll:
            fout.write('%s\n' % elem)

def load_collection(infile):
    with open(infile) as fin:
        coll = [elem.strip() for elem in fin.readlines()]
    return coll

def load_sth_to_id(infile):
    with open(infile) as fin:
        sth_list = [sth.strip() for sth in fin.readlines()]
    sth_to_id = dict(zip(sth_list, range(len(sth_list))))
    return sth_to_id

def load_label_to_id():
    label_to_id = load_sth_to_id(config.label_file)
    return label_to_id

def load_token_to_id():
    vocab_to_id = load_sth_to_id(config.vocab_file)
    return vocab_to_id

def load_id_to_sth(infile):
    with open(infile) as fin:
        sth_list = [sth.strip() for sth in fin.readlines()]
    id_to_sth = dict(zip(range(len(sth_list)), sth_list))
    return id_to_sth

def load_id_to_label():
    id_to_label = load_id_to_sth(config.label_file)
    return id_to_label

def load_id_to_token():
    id_to_vocab = load_id_to_sth(config.vocab_file)
    return id_to_vocab

def count_data_size(infile):
    with open(infile) as fin:
        data = [line.strip() for line in fin.readlines()]
    data_size = len(data)
    return data_size

def decode_tfrecord_bak(tfrecord_file, shuffle=True):
    Tensor = slim.tfexample_decoder.Tensor
    Image = slim.tfexample_decoder.Image
    TFExampleDecoder = slim.tfexample_decoder.TFExampleDecoder
    Dataset = slim.dataset.Dataset
    DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider

    data_sources = [tfrecord_file]
    num_label = config.num_label
    token_to_id = load_token_to_id()
    unk_token_id = token_to_id[config.unk_token]
    reader = tf.TFRecordReader
    keys_to_features = {
        config.user_key:tf.FixedLenFeature((), tf.string,
                default_value=''),
        config.image_encoded_key:tf.FixedLenFeature((), tf.string,
                default_value=''),
        config.text_key:tf.VarLenFeature(dtype=tf.int64),
        config.label_key:tf.FixedLenFeature([num_label], tf.int64,
                default_value=tf.zeros([num_label], dtype=tf.int64)),
        config.image_format_key:tf.FixedLenFeature((), tf.string,
                default_value='jpg'),
        config.image_file_key:tf.FixedLenFeature((), tf.string,
                default_value='')
    }
    items_to_handlers = {
        'user':Tensor(config.user_key),
        'image':Image(),
        'text':Tensor(config.text_key, default_value=unk_token_id),
        'label':Tensor(config.label_key),
        'image_file':Tensor(config.image_file_key),
    }
    decoder = TFExampleDecoder(keys_to_features, items_to_handlers)
    num_samples = np.inf
    items_to_descriptions = {
        'user':'',
        'image':'',
        'text':'',
        'label':'',
        'image_file':'',
    }
    dataset = Dataset(
        data_sources=data_sources,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
    )
    provider = DatasetDataProvider(dataset, shuffle=shuffle)
    ts_list = provider.get(['user', 'image', 'text', 'label', 'image_file'])
    return ts_list

def generate_batch_bak(model, ts_list, batch_size):
    get_preprocessing = preprocessing_factory.get_preprocessing
    preprocessing = get_preprocessing(model.preprocessing_name,
            is_training=model.is_training)
    user_ts, image_ts, text_ts, label_ts, image_file_ts = ts_list
    image_ts = preprocessing(image_ts, model.image_size, model.image_size)
    user_bt, image_bt, text_bt, label_bt, image_file_bt = tf.train.batch(
            [user_ts, image_ts, text_ts, label_ts, image_file_ts], 
            batch_size=batch_size,
            dynamic_pad=True,
            num_threads=config.num_threads)
    return user_bt, image_bt, text_bt, label_bt, image_file_bt

def get_data_sources(flags, is_training=True, single=False):
  for (dirpath, dirnames, filenames) in os.walk(config.prerecord_dir):
    break
  marker = 'train'
  if not is_training:
    marker = 'valid'
  data_sources = []
  for filename in filenames:
    if filename.find(marker) < 0:
      continue
    if filename.find(flags.model_name) < 0:
      continue
    if single and (filename.find('000') < 0):
      continue
    filepath = path.join(config.prerecord_dir, filename)
    data_sources.append(filepath)
  return data_sources

def decode_tfrecord(flags, data_sources, shuffle=True):
    Tensor = slim.tfexample_decoder.Tensor
    TFExampleDecoder = slim.tfexample_decoder.TFExampleDecoder
    Dataset = slim.dataset.Dataset
    DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider

    num_label = config.num_label
    token_to_id = load_token_to_id()
    unk_token_id = token_to_id[config.unk_token]
    reader = tf.TFRecordReader
    keys_to_features = {
        config.user_key:tf.FixedLenFeature((), tf.string),
        config.image_key:tf.FixedLenFeature([flags.feature_size], tf.float32),
        config.text_key:tf.VarLenFeature(dtype=tf.int64),
        config.label_key:tf.FixedLenFeature([num_label], tf.int64),
        config.file_key:tf.FixedLenFeature((), tf.string)
    }
    items_to_handlers = {
        'user':Tensor(config.user_key),
        'image':Tensor(config.image_key),
        'text':Tensor(config.text_key, default_value=unk_token_id),
        'label':Tensor(config.label_key),
        'file':Tensor(config.file_key),
    }
    decoder = TFExampleDecoder(keys_to_features, items_to_handlers)
    num_samples = np.inf
    items_to_descriptions = {
        'user':'',
        'image':'',
        'text':'',
        'label':'',
        'file':'',
    }
    dataset = Dataset(
        data_sources=data_sources,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
    )
    provider = DatasetDataProvider(dataset, shuffle=shuffle)
    ts_list = provider.get(['user', 'image', 'text', 'label', 'file'])
    return ts_list

def generate_batch(ts_list, batch_size):
    user_ts, image_ts, text_ts, label_ts, file_ts = ts_list
    label_ts = tf.divide(label_ts, tf.reduce_sum(label_ts))
    user_bt, image_bt, text_bt, label_bt, file_bt = tf.train.batch(
            [user_ts, image_ts, text_ts, label_ts, file_ts], 
            batch_size=batch_size,
            dynamic_pad=True,
            num_threads=config.num_threads)
    return user_bt, image_bt, text_bt, label_bt, file_bt

def evaluate(flags, sess, gen_v, bt_list_v):
  num_batch_v = int(config.valid_data_size / config.valid_batch_size)
  # print('vd:\t#batch=%d\n' % num_batch_v)
  user_bt_v, image_bt_v, text_bt_v, label_bt_v, file_bt_v = bt_list_v
  image_hit_v = []
  for batch_v in range(num_batch_v):
    image_np_v, label_np_v = sess.run([image_bt_v, label_bt_v])
    feed_dict = {gen_v.image_ph:image_np_v}
    
    image_logit_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
    image_hit_bt = metric.compute_hit(image_logit_v, label_np_v, flags.cutoff)
    image_hit_v.append(image_hit_bt)
  image_hit_v = np.mean(image_hit_v)
  return image_hit_v










