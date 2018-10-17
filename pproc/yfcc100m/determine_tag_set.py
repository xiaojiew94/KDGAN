from nltk.corpus import wordnet
from urllib import request

import string
import calendar
import pickle
import pycountry
import re

def is_imagenet_tag(word):
  if len(word) < 3:
    return False
  return True

def save_wordnet_tag_set():
  wordnet_file = 'wordnet_tag_set.p'
  wordnet_tag_set = set()

  pickle.dump(wordnet_tag_set, open(wordnet_file, 'wb'))

def save_imagenet_tag_set():
  imagenet_file = 'imagenet_tag_set.p'
  imagenet_tag_set = set()

  base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
  synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
  synset_to_name_url = '{}/imagenet_metadata.txt'.format(base_url)
  filename, _ = request.urlretrieve(synset_url)
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synset = len(synset_list)
  assert num_synset == 1000
  filename, _ = request.urlretrieve(synset_to_name_url)
  synset_to_text_list = open(filename).readlines()
  tot_synset = len(synset_to_text_list)
  assert tot_synset == 21842
  synset_to_name = {}
  for s in synset_to_text_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    text = parts[1]
    synset_to_name[synset] = text
  word_set = set()
  label_id = 0
  for synset in synset_list:
    text = synset_to_name[synset].lower()
    word_list = re.findall(r'[a-zA-Z-]+', text)
    label_id += 1
    # print('#%04d %s' % (label_id, text))
    word_set = word_set.union(word_list)

  for word in word_set:
    if is_imagenet_tag(word):
      imagenet_tag_set.add(word)
  for tag in imagenet_tag_set:
    print(tag)

  pickle.dump(imagenet_tag_set, open(imagenet_file, 'wb'))

def main():
  save_wordnet_tag_set()
  save_imagenet_tag_set()

if __name__ == '__main__':
  main()
