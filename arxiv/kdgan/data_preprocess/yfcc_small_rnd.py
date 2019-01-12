from kdgan import config
from kdgan import utils

import operator
import os
import random
import shutil
import string
import urllib

import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from datasets import imagenet
from nets import nets_factory
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from os import path
from preprocessing import preprocessing_factory
from sys import stdout
from tensorflow.contrib import slim

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from datasets.download_and_convert_flowers import ImageReader
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from PIL import Image

tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
tf.app.flags.DEFINE_string('end_point', None, '')
tf.app.flags.DEFINE_string('pretrained_ckpt', None, '')
tf.app.flags.DEFINE_integer('channels', 3, '')
tf.app.flags.DEFINE_boolean('overwrite', False, '')
tf.app.flags.DEFINE_boolean('baseline', False, '')
flags = tf.app.flags.FLAGS

lemmatizer = WordNetLemmatizer()

SPACE_PLUS = '+'
LABEL_SEPERATOR = ','
POST_INDEX = 0
USER_INDEX = 1
IMAGE_INDEX = 2
TEXT_INDEX = 3
DESC_INDEX = 4
LABEL_INDEX = -1

FIELD_SEPERATOR = '\t'
EXPECTED_NUM_FIELD = 6

MIN_RND_LABEL = 10
NUM_RND_LABEL = 250
MIN_RND_POST = MIN_RND_LABEL
NUM_RND_POST = 10000
TRAIN_DATA_RATIO = 0.80
SHUFFLE_SEED = 100

dataset = 'yfcc_rnd'
dataset_dir = config.yfcc_rnd_dir
utils.create_if_nonexist(dataset_dir)
raw_file = path.join(dataset_dir, '%s.raw' % dataset)
data_file = path.join(dataset_dir, '%s.data' % dataset)
train_file = path.join(dataset_dir, '%s.train' % dataset)
valid_file = path.join(dataset_dir, '%s.valid' % dataset)
label_file = path.join(dataset_dir, '%s.label' % dataset)
vocab_file = path.join(dataset_dir, '%s.vocab' % dataset)
image_data_dir = path.join(dataset_dir, 'ImageData')

################################################################
#
# create kdgan data
#
################################################################

def check_num_field():
  fin = open(config.sample_file)
  while True:
    line = fin.readline()
    if not line:
      # print('line=\'{}\' type={}'.format(line, type(line)))
      break
    fields = line.strip().split(FIELD_SEPERATOR)
    num_field = len(fields)
    if num_field != EXPECTED_NUM_FIELD:
      raise Exception('wrong number of fields')
  fin.close()

def is_noun(label):
  is_noun = True
  synsets = wordnet.synsets(label)
  for synset in synsets:
    lexical_type = synset.lexname()
    lexical_type = lexical_type.split('.')[0]
    if lexical_type != 'noun':
      is_noun = False
      break
  return is_noun

def select_rnd_label():
  imagenet_labels = {}
  label_names = imagenet.create_readable_names_for_imagenet_labels()
  label_names = {k:v.lower() for k, v in label_names.items()}
  for label_id in range(1, 1001):
    names = label_names[label_id]
    for name in names.split(','):
      name = name.strip()
      label = name.split()[-1]
      if label not in imagenet_labels:
        imagenet_labels[label] = []
      imagenet_labels[label].append(names)

  fin = open(config.sample_file)
  label_count = {}
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    labels = fields[LABEL_INDEX]
    labels = labels.split(LABEL_SEPERATOR)
    for label in labels:
      label_count[label] = label_count.get(label, 0) + 1
  fin.close()

  invalid_labels = [
    # people
    'admiral',
    'beauty',
    'feeder',
    'finch',
    'harvestman',
    'jack',
    'maker',
    'maria',
    'mike',
    'monarch',
    'monster',
    'painter',
    'player',
    'robin',
    'smith',
    'speaker',
    # abstract
    'newfoundland',
    'pickup',
    'scope',
  ]
  valid_labels = {}
  for label, count in label_count.items():
    if label in invalid_labels:
      continue
    if label not in imagenet_labels:
      continue
    if count < MIN_RND_LABEL:
      continue
    if not is_noun(label):
      continue
    valid_labels[label] = count
  valid_labels = sorted(valid_labels.items(), key=operator.itemgetter(1))
  rnd_labels = [label for label, count in valid_labels]
  print('\t#label=%d/%d' % (NUM_RND_LABEL, len(rnd_labels)))
  rnd_labels = np.random.choice(rnd_labels, size=NUM_RND_LABEL, replace=False)
  rnd_labels = sorted(rnd_labels)
  # for count, label in enumerate(rnd_labels):
  #   names = []
  #   for label_id in range(1, 1001):
  #     if label in label_names[label_id]:
  #       names.append(label_names[label_id])
  #   print('#%d label=%s' % (count + 1, label))
  #   for names in imagenet_labels[label]:
  #     print('\t%s' %(names))
  #   input()
  utils.save_collection(rnd_labels, label_file)

def with_rnd_label(labels, rnd_labels):
  old_labels = labels.split(LABEL_SEPERATOR)
  new_labels = []
  for label in old_labels:
    if label not in rnd_labels:
      continue
    new_labels.append(label)
  if len(new_labels) == 0:
    return False
  return True

def keep_rnd_label(labels, rnd_labels):
  old_labels = labels.split(LABEL_SEPERATOR)
  new_labels = []
  for label in old_labels:
    if label not in rnd_labels:
      continue
    new_labels.append(label)
  return new_labels

def save_posts(posts, outfile):
  image_set = set()
  with open(outfile, 'w') as fout:
    posts = sorted(posts, key=operator.itemgetter(0))
    for post in posts:
      fields = post.split(FIELD_SEPERATOR)
      image = fields[IMAGE_INDEX]
      image_set.add(image)
      fout.write('%s\n' % post)

def get_labels(post):
  fields = post.split(FIELD_SEPERATOR)
  user = fields[USER_INDEX]
  labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
  return labels

def get_label_count(posts):
  label_count = {}
  for post in posts:
    labels = get_labels(post)
    for label in labels:
      label_count[label] = label_count.get(label, 0) + 1
  return label_count

def sample_posts(in_posts):
  label_count = {}
  out_posts, rem_posts = [], []
  for post in in_posts:
    labels = get_labels(post)
    skip = True
    for label in labels:
      count = label_count.get(label, 0)
      if count < MIN_RND_POST:
        skip = False
        break
    if skip:
      rem_posts.append(post)
      continue
    out_posts.append(post)
    for label in labels:
      count = label_count.get(label, 0)
      label_count[label] = count + 1
  out_post = len(out_posts)
  # print('\t#in=%d #cur=%d rem=%d' % (len(in_posts), out_post, len(rem_posts)))
  rnd_post = NUM_RND_POST - out_post
  rnd_posts = np.random.choice(rem_posts, size=rnd_post, replace=False)
  out_posts.extend(rnd_posts)
  print('\t#post=%d/%d' % (len(out_posts), len(in_posts)))
  return out_posts

def select_posts():
  rnd_labels = utils.load_collection(label_file)

  posts = []
  fin = open(config.sample_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    user, labels = fields[USER_INDEX], fields[LABEL_INDEX]
    image_url = fields[IMAGE_INDEX]
    if not with_rnd_label(labels, rnd_labels):
      continue
    post = fields[POST_INDEX]
    if post in []:
      continue
    labels = keep_rnd_label(labels, rnd_labels)
    fields[LABEL_INDEX] = LABEL_SEPERATOR.join(labels)
    image_file = path.basename(image_url)
    image = image_file.split('_')[0]
    fields[IMAGE_INDEX] = image
    posts.append(FIELD_SEPERATOR.join(fields))
  fin.close()

  posts = sample_posts(posts)
  label_count = get_label_count(posts)
  num_label = len(label_count)
  counts = label_count.values()
  min_count, max_count = min(counts), max(counts)
  print('\t#label=%d [%d, %d]' % (num_label, min_count, max_count))

  save_posts(posts, raw_file)
  return min(counts)

stopwords = set(stopwords.words('english'))
def tokenize_dataset():
  stemmer = SnowballStemmer('english')
  tokenizer = RegexpTokenizer('[a-z]+')
  def _in_wordnet(token):
    if wordnet.synsets(token):
      if any(c not in string.ascii_lowercase + '-' for c in token):
        return False
      if len(token) < 3:
        return False
      return True
    else:
      return False
  def _stop_stem(tokens):
    tokens = [token for token in tokens if _in_wordnet(token)]
    tokens = [token for token in tokens if not token in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

  fin = open(raw_file)
  fout = open(data_file, 'w')
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    post = fields[POST_INDEX]
    text = fields[TEXT_INDEX]
    desc = fields[DESC_INDEX]
    text = ' '.join([text, desc])

    text = urllib.parse.unquote(text)
    text = text.replace(SPACE_PLUS, ' ')

    soup = BeautifulSoup(text, 'html.parser')
    children = []
    for child in soup.children:
      if type(child) == NavigableString:
        children.append(str(child))
      else:
        children.append(str(child.text))
    text = ' '.join(children)
    tokens = word_tokenize(text)
    tokens = _stop_stem(tokens)
    if len(tokens) == 0:
      tokens = tokenizer.tokenize(text)
      tokens = _stop_stem(tokens)
      if len(tokens) == 0:
        tokens = tokenizer.tokenize(text)
    if len(tokens) == 0:
      raise Exception('no textual information', text)
    text = ' '.join(tokens)

    labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
    labels = ' '.join(labels)
    fields = [fields[POST_INDEX], fields[USER_INDEX], fields[IMAGE_INDEX]]
    fields.extend([text, labels])
    fout.write('%s\n' % FIELD_SEPERATOR.join(fields))
  fout.close()
  fin.close()

def check_dataset(infile):
  rnd_labels = utils.load_collection(label_file)
  rnd_labels = set(rnd_labels)
  fin = open(infile)
  while True:
    line = fin.readline()
    if not line:
      break
    fields = line.strip().split(FIELD_SEPERATOR)
    labels = fields[LABEL_INDEX].split()
    for label in labels:
      rnd_labels.discard(label)
  print(path.basename(infile), len(rnd_labels))
  assert len(rnd_labels) == 0

def split_dataset():
  posts = []
  fin = open(data_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    posts.append(line)
  fin.close()
  train_posts, valid_posts = [], []
  random.seed(SHUFFLE_SEED)
  num_post = len(posts)
  seed = random.random()
  random.shuffle(posts, lambda:seed)

  valid_post = []
  valid_label_count = {}
  rem_posts = []
  for post in posts:
    labels = get_labels(post)
    skip = True
    for label in labels:
      count = valid_label_count.get(label, 0)
      if count == 0:
        skip = False
        break
    if skip:
      rem_posts.append(post)
      continue
    valid_posts.append(post)
    for label in labels:
      count = valid_label_count.get(label, 0)
      valid_label_count[label] = count + 1
  seperator = int(num_post * TRAIN_DATA_RATIO)
  train_posts = rem_posts[:seperator]
  valid_posts.extend(rem_posts[seperator:])

  save_posts(train_posts, train_file)
  save_posts(valid_posts, valid_file)

  check_dataset(train_file)
  check_dataset(valid_file)

  vocab = set()
  for post in train_posts:
    fields = post.split(FIELD_SEPERATOR)
    tokens = fields[TEXT_INDEX].split()
    for token in tokens:
      vocab.add(token)
  vocab = sorted(vocab)
  if config.unk_token in vocab:
    raise Exception('please change unk token', config.unk_token)
  vocab.insert(0, config.unk_token)
  if config.pad_token in vocab:
    raise Exception('please change pad token', config.pad_token)
  vocab.insert(0, config.pad_token)
  utils.save_collection(vocab, vocab_file)

def get_image_path(image_dir, image_url):
  fields = image_url.split('/')
  image_path = path.join(image_dir, fields[-2], fields[-1])
  return image_path

def collect_image(infile, outdir):
  post_image = {}
  fin = open(infile)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    post = fields[POST_INDEX]
    image = fields[IMAGE_INDEX]
    post_image[post] = image
  fin.close()
  utils.create_if_nonexist(outdir)
  fin = open(config.sample_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    post = fields[POST_INDEX]
    if post not in post_image:
      continue
    image_url = fields[IMAGE_INDEX]
    src_file = get_image_path(config.image_dir, image_url)
    image = post_image[post]
    dst_file = path.join(outdir, '%s.jpg' % image)
    if path.isfile(dst_file):
      continue
    shutil.copyfile(src_file, dst_file)

################################################################
#
# create survey data
#
################################################################

def survey_image_data(infile, dataset):
  image_data = path.join(config.surv_dir, dataset, 'ImageData')
  utils.create_if_nonexist(image_data)
  outfile = path.join(image_data, '%s.txt' % dataset)
  fout = open(outfile, 'w')
  fin = open(infile)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image = fields[IMAGE_INDEX]
    image_file = '%s.jpg' % image
    fout.write('{}\n'.format(image_file))
  fin.close()
  fout.close()
  collect_image(infile, image_data)

def survey_text_data(infile, dataset):
  seperator = '###'
  def _get_key(label_i, label_j):
    if label_i < label_j:
      key = label_i + seperator + label_j
    else:
      key = label_j + seperator + label_i
    return key
  def _get_labels(key):
    fields = key.split(seperator)
    label_i, label_j = fields[0], fields[1]
    return label_i, label_j

  text_data = path.join(config.surv_dir, dataset, 'TextData')
  utils.create_if_nonexist(text_data)

  post_image = {}
  fin = open(infile)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    post = fields[POST_INDEX]
    image = fields[IMAGE_INDEX]
    post_image[post] = image
  fin.close()

  rawtags_file = path.join(text_data, 'id.userid.rawtags.txt')
  fout = open(rawtags_file, 'w')
  fin = open(config.rawtag_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    post = fields[0]
    if post not in post_image:
      continue
    post = fields[POST_INDEX]
    image = post_image[post]
    user = fields[USER_INDEX]
    old_labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
    new_labels = []
    for old_label in old_labels:
      old_label = urllib.parse.unquote(old_label)
      old_label = old_label.lower()
      new_label = ''
      for c in old_label:
        if not c.isalnum():
          continue
        new_label += c
      if len(new_label) == 0:
        continue
      new_labels.append(new_label)
    labels = ' '.join(new_labels)
    fout.write('{}\t{}\t{}\n'.format(image, user, labels))
  fin.close()
  fout.close()

  lemmtags_file = path.join(text_data, 'id.userid.lemmtags.txt')
  fout = open(lemmtags_file, 'w')
  fin = open(rawtags_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    old_labels = fields[-1].split(' ')
    new_labels = []
    for old_label in old_labels:
      new_label = lemmatizer.lemmatize(old_label)
      new_labels.append(new_label)
    fields[-1] = ' '.join(new_labels)
    fout.write('{}\n'.format(FIELD_SEPERATOR.join(fields)))
  fin.close()
  fout.close()

  fin = open(lemmtags_file)
  label_users, label_images = {}, {}
  label_set = set()
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image, user = fields[0], fields[1]
    labels = fields[2].split()
    for label in labels:
      if label not in label_users:
        label_users[label] = set()
      label_users[label].add(user)
      if label not in label_images:
        label_images[label] = set()
      label_images[label].add(image)
      label_set.add(label)
  fin.close()
  tagfreq_file = path.join(text_data, 'lemmtag.userfreq.imagefreq.txt')
  fout = open(tagfreq_file, 'w')
  label_count = {}
  for label in label_set:
    label_count[label] = len(label_users[label]) # + len(label_images[label])
  sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
  for label, _ in sorted_label_count:
    userfreq = len(label_users[label])
    imagefreq = len(label_images[label])
    fout.write('{} {} {}\n'.format(label, userfreq, imagefreq))
  fout.close()

  jointfreq_file = path.join(text_data, 'ucij.uuij.icij.iuij.txt')
  min_count = 4
  if not infile.endswith('.valid'):
    min_count = 8
  label_count = {}
  fin = open(lemmtags_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image, user = fields[0], fields[1]
    labels = fields[2].split()
    for label in labels:
      if label not in label_count:
        label_count[label] = 0
      label_count[label] += 1
  fin.close()
  jointfreq_icij_init = {}
  fin = open(lemmtags_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image, user = fields[0], fields[1]
    labels = fields[2].split()
    num_label = len(labels)
    for i in range(num_label - 1):
      for j in range(i + 1, num_label):
        label_i = labels[i]
        label_j = labels[j]
        if label_i == label_j:
          continue
        if label_count[label_i] < min_count:
          continue
        if label_count[label_j] < min_count:
          continue
        key = _get_key(label_i, label_j)
        if key not in jointfreq_icij_init:
          jointfreq_icij_init[key] = 0
        jointfreq_icij_init[key] += 1
  fin.close()
  keys = set()
  icij_images = {}
  iuij_images = {}
  fin = open(lemmtags_file)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image, user = fields[0], fields[1]
    labels = fields[2].split()
    num_label = len(labels)
    for i in range(num_label - 1):
      for j in range(i + 1, num_label):
        label_i = labels[i]
        label_j = labels[j]
        if label_i == label_j:
          continue
        if label_i not in iuij_images:
          iuij_images[label_i] = set()
        iuij_images[label_i].add(image)
        if label_j not in iuij_images:
          iuij_images[label_j] = set()
        iuij_images[label_j].add(image)
        if label_count[label_i] < min_count:
          continue
        if label_count[label_j] < min_count:
          continue
        key = _get_key(label_i, label_j)
        if jointfreq_icij_init[key] < min_count:
          continue
        keys.add(key)
        if key not in icij_images:
          icij_images[key] = set()
        icij_images[key].add(image)
  fin.close()
  jointfreq_icij, jointfreq_iuij = {}, {}
  keys = sorted(keys)
  for key in keys:
    jointfreq_icij[key] = len(icij_images[key])
    label_i, label_j = _get_labels(key)
    label_i_images = iuij_images[label_i]
    label_j_images = iuij_images[label_j]
    jointfreq_iuij[key] = len(label_i_images.union(label_j_images))
  fout = open(jointfreq_file, 'w')
  for key in sorted(keys):
    label_i, label_j = _get_labels(key)
    fout.write('{} {} {} {} {} {}\n'.format(label_i, label_j, jointfreq_icij[key], jointfreq_iuij[key], jointfreq_icij[key], jointfreq_iuij[key]))
  fout.close()

  fin = open(lemmtags_file)
  vocab = set()
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image, user = fields[0], fields[1]
    labels = fields[2].split()
    for label in labels:
      if wordnet.synsets(label):
        vocab.add(label)
      else:
        pass
  fin.close()
  vocab_file = path.join(text_data, 'wn.%s.txt' % dataset)
  fout = open(vocab_file, 'w')
  for label in sorted(vocab):
    fout.write('{}\n'.format(label))
  fout.close()

def survey_feature_sets(infile, dataset):
  image_sets = path.join(config.surv_dir, dataset, 'ImageSets')
  utils.create_if_nonexist(image_sets)

  outfile = path.join(image_sets, '%s.txt' % dataset)
  fout = open(outfile, 'w')
  fin = open(infile)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image = fields[IMAGE_INDEX]
    fout.write('{}\n'.format(image))
  fin.close()
  fout.close()

  fout = open(path.join(image_sets, 'holdout.txt'), 'w')
  fout.close()

def survey_annotations(infile, dataset):
  annotations = path.join(config.surv_dir, dataset, 'Annotations')
  utils.create_if_nonexist(annotations)
  concepts = 'concepts.txt'
  
  label_set = set()
  label_images = {}
  image_set = set()
  fin = open(infile)
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    image = fields[IMAGE_INDEX]
    labels = fields[LABEL_INDEX].split()
    for label in labels:
      label_set.add(label)
      if label not in label_images:
        label_images[label] = []
      label_images[label].append(image)
    image_set.add(image)
  fin.close()
  fout = open(path.join(annotations, concepts), 'w')
  for label in sorted(label_set):
    fout.write('{}\n'.format(label))
  fout.close()

  concepts_dir = path.join(annotations, 'Image', concepts)
  utils.create_if_nonexist(concepts_dir)
  image_list = sorted(image_set)
  for label in label_set:
    label_filepath = path.join(concepts_dir, '%s.txt' % label)
    fout = open(label_filepath, 'w')
    for image in image_list:
      assessment = -1
      if image in label_images[label]:
        assessment = 1
      fout.write('{} {}\n'.format(image, assessment))
    fout.close()

def create_survey_data():
  train_dataset = '%s_tn' % dataset
  valid_dataset = '%s_vd' % dataset
  surv_train_dir = path.join(config.surv_dir, train_dataset)
  surv_valid_dir = path.join(config.surv_dir, valid_dataset)

  print('remove survey train data')
  utils.delete_if_exist(surv_train_dir)
  print('remove survey valid data')
  utils.delete_if_exist(surv_valid_dir)

  print('collect survey train images')
  survey_image_data(train_file, train_dataset)
  print('collect survey valid images')
  survey_image_data(valid_file, valid_dataset)

  print('create survey train text data')
  survey_text_data(train_file, train_dataset)
  print('create survey valid text data')
  survey_text_data(valid_file, valid_dataset)

  print('create survey train feature set')
  survey_feature_sets(train_file, train_dataset)
  print('create survey valid feature set')
  survey_feature_sets(valid_file, valid_dataset)

  print('create survey train annotations')
  survey_annotations(train_file, train_dataset)
  print('create survey valid annotations')
  survey_annotations(valid_file, valid_dataset)

################################################################
#
# use pretrained model
#
################################################################

num_classes = 1000
if flags.model_name not in ['vgg_16', 'vgg_19']:
    num_classes = 1001
# print('#class=%d' % (num_classes))
network_fn_t = nets_factory.get_network_fn(flags.model_name,
        num_classes=num_classes,
        is_training=True)
network_fn_v = nets_factory.get_network_fn(flags.model_name,
        num_classes=num_classes,
        is_training=False)
image_size_t = network_fn_t.default_image_size
image_size_v = network_fn_v.default_image_size
assert image_size_t==image_size_v
image_size = int((image_size_t + image_size_v) / 2)
# print('image size=%d' % (image_size))
image_ph = tf.placeholder(tf.float32, shape=(None, None, flags.channels))
preprocessing_t = preprocessing_factory.get_preprocessing(flags.preprocessing_name,
        is_training=True)
preprocessing_v = preprocessing_factory.get_preprocessing(flags.preprocessing_name,
        is_training=False)
image_ts_t = tf.expand_dims(preprocessing_t(image_ph, image_size, image_size),
        axis=0)
image_ts_v = tf.expand_dims(preprocessing_v(image_ph, image_size, image_size),
        axis=0)
_, end_points_t = network_fn_t(image_ts_t)
scope = tf.get_variable_scope()
scope.reuse_variables()
_, end_points_v = network_fn_v(image_ts_v)
end_point_t = tf.squeeze(end_points_t[flags.end_point])
# print('tn', end_point_t.shape, end_point_t.dtype)
end_point_v = tf.squeeze(end_points_v[flags.end_point])
# print('vd', end_point_v.shape, end_point_v.dtype)

# print('trainable parameters')
# for variable in slim.get_model_variables():
#     num_params = 1
#     for dim in variable.shape:
#         num_params *= dim.value
#     print('\t', variable.name, '\t', num_params)
# print('end points')
# for name, tensor in end_points_t.items():
#     print('\t', name, '\t', tensor.shape)

variables_to_restore = slim.get_variables_to_restore()
init_fn = slim.assign_from_checkpoint_fn(flags.pretrained_ckpt, variables_to_restore)

def build_example(user, image, text, label, file):
    return tf.train.Example(features=tf.train.Features(feature={
        user_key:dataset_utils.bytes_feature(user),
        image_key:dataset_utils.float_feature(image),
        text_key:dataset_utils.int64_feature(text),
        label_key:dataset_utils.int64_feature(label),
        file_key:dataset_utils.bytes_feature(file),
    }))

def create_tfrecord(infile, end_point, is_training=False):
    utils.create_if_nonexist(precomputed_dir)

    num_epoch = flags.num_epoch
    if not is_training:
        num_epoch = 1

    fields = path.basename(infile).split('.')
    dataset, version = fields[0], fields[1]
    filepath = path.join(precomputed_dir, tfrecord_tmpl)

    user_list = []
    file_list = []
    text_list = []
    label_list = []
    fin = open(infile)
    while True:
        line = fin.readline()
        if not line:
            break
        fields = line.strip().split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        image = fields[IMAGE_INDEX]
        file = path.join(image_data_dir, '%s.jpg' % image)
        tokens = fields[TEXT_INDEX].split()
        labels = fields[LABEL_INDEX].split()
        user_list.append(user)
        file_list.append(file)
        text_list.append(tokens)
        label_list.append(labels)
    fin.close()

    label_to_id = utils.load_sth_to_id(label_file)
    num_label = len(label_to_id)
    print('#label={}'.format(num_label))
    token_to_id = utils.load_sth_to_id(vocab_file)
    unk_token_id = token_to_id[config.unk_token]
    vocab_size = len(token_to_id)
    print('#vocab={}'.format(vocab_size))

    reader = ImageReader()
    with tf.Session() as sess:
        init_fn(sess)
        for epoch in range(num_epoch):
            count = 0
            tfrecord_file = filepath.format(dataset, flags.model_name, epoch, version)
            if path.isfile(tfrecord_file):
                continue
            # print(tfrecord_file)
            # exit()
            with tf.python_io.TFRecordWriter(tfrecord_file) as fout:
                for user, file, text, labels in zip(user_list, file_list, text_list, label_list):
                    user = bytes(user, encoding='utf-8')
                    
                    image_np = np.array(Image.open(file))
                    # print(type(image_np), image_np.shape)
                    feed_dict = {image_ph:image_np}
                    image, = sess.run([end_point], feed_dict)
                    image = image.tolist()
                    # print(image)
                    # print(type(image), len(image))
                    # input()

                    text = [token_to_id.get(token, unk_token_id) for token in text]

                    label_ids = [label_to_id[label] for label in labels]
                    label_vec = np.zeros((num_label,), dtype=np.int64)
                    label_vec[label_ids] = 1
                    label = label_vec.tolist()

                    file = bytes(file, encoding='utf-8')
                    # print(file)

                    example = build_example(user, image, text, label, file)
                    fout.write(example.SerializeToString())
                    count += 1
                    if (count % 500) == 0:
                        print('count={}'.format(count))

def create_test_set():
    utils.create_if_nonexist(precomputed_dir)

    user_list = []
    file_list = []
    text_list = []
    label_list = []
    fin = open(valid_file)
    valid_size = 0
    while True:
        line = fin.readline()
        if not line:
            break
        fields = line.strip().split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        image = fields[IMAGE_INDEX]
        file = path.join(image_data_dir, '%s.jpg' % image)
        tokens = fields[TEXT_INDEX].split()
        labels = fields[LABEL_INDEX].split()
        user_list.append(user)
        file_list.append(file)
        text_list.append(tokens)
        label_list.append(labels)
        valid_size += 1
    fin.close()

    label_to_id = utils.load_sth_to_id(label_file)
    num_label = len(label_to_id)
    print('#label={}'.format(num_label))
    token_to_id = utils.load_sth_to_id(vocab_file)
    unk_token_id = token_to_id[config.unk_token]
    vocab_size = len(token_to_id)
    print('#vocab={}'.format(vocab_size))

    image_npy = np.zeros((valid_size, 4096), dtype=np.float32)
    label_npy = np.zeros((valid_size, 100), dtype=np.int32)
    imgid_npy = []
    text_npy = []
    reader = ImageReader()
    with tf.Session() as sess:
        init_fn(sess)
        for i, (user, file, text, labels) in enumerate(zip(user_list, file_list, text_list, label_list)):
            user = bytes(user, encoding='utf-8')
            
            image_np = np.array(Image.open(file))
            # print(type(image_np), image_np.shape)
            feed_dict = {image_ph:image_np}
            image, = sess.run([end_point_v], feed_dict)
            image = image.tolist()
            # print(image)
            # print(type(image), len(image))
            image_npy[i,:] = image
            # print(image_npy)
            # input()

            text = [token_to_id.get(token, unk_token_id) for token in text]
            text_npy.append(text)

            label_ids = [label_to_id[label] for label in labels]
            label_vec = np.zeros((num_label,), dtype=np.int32)
            label_vec[label_ids] = 1
            label = label_vec.tolist()
            label_npy[i,:] = label

            image_id = path.basename(file).split('.')[0]
            imgid_npy.append(image_id)
            # example = build_example(user, image, text, label, file)

    imgid_npy = np.asarray(imgid_npy)
    filename_tmpl = 'yfcc10k_%s.valid.%s'
    np.save(path.join(precomputed_dir, filename_tmpl % (flags.model_name, 'image')), image_npy)
    np.save(path.join(precomputed_dir, filename_tmpl % (flags.model_name, 'label')), label_npy)
    np.save(path.join(precomputed_dir, filename_tmpl % (flags.model_name, 'imgid')), imgid_npy)

    def padding(x):
        x = np.array(x)
        print(x.shape)
        max_length = max(len(row) for row in x)
        x = np.array([row + [0] * (max_length - len(row)) for row in x])
        print(x.shape)
        return x
    text_npy = padding(text_npy)
    np.save(path.join(precomputed_dir, filename_tmpl % (flags.model_name, 'text')), text_npy)

def main(_):
  if flags.overwrite:
    print('create yfcc small rnd dataset')
    utils.delete_if_exist(dataset_dir)
    utils.create_if_nonexist(dataset_dir)

  check_num_field()
  if flags.overwrite or (not utils.skip_if_exist(raw_file)):
    while True:
      print('random labels and posts')
      select_rnd_label()
      min_count = select_posts()
      if min_count < MIN_RND_POST:
        continue
      break

  if flags.overwrite or (not utils.skip_if_exist(data_file)):
    print('tokenize and collect images')
    tokenize_dataset()
    collect_image(data_file, image_data_dir)

  if (flags.overwrite
        or not utils.skip_if_exist(train_file)
        or not utils.skip_if_exist(valid_file)
        or not utils.skip_if_exist(vocab_file)):
    while True:
      print('split into train and valid')
      try:
        split_dataset()
        break
      except:
        continue

  if flags.baseline:
    print('create survey data')
    create_survey_data()

  # create_test_set()
  # create_tfrecord(valid_file, end_point_v, is_training=False)
  # create_tfrecord(train_file, end_point_t, is_training=True)

if __name__ == '__main__':
  tf.app.run()