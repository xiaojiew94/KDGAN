import utils

from nltk.corpus import wordnet
from sys import stdout
from urllib import request

import pickle
import pycountry
import re

excluded_file = 'imagenet_excluded.txt'
def sort_excluded():
  imagenet_excl = set()
  with open(excluded_file) as fin:
    for line in fin.readlines():
      imagenet_excl.add(line.strip())
  with open(excluded_file, 'w') as fout:
    for excl in sorted(imagenet_excl):
      fout.write('%s\n' % (excl))
sort_excluded()

def get_imagenet_excl():
  imagenet_excl = set()

  for country in pycountry.countries:
    imagenet_excl.add(country.name.lower())
  for country in pycountry.historic_countries:
    imagenet_excl.add(country.name.lower())
  for subdivision in pycountry.subdivisions:
    imagenet_excl.add(subdivision.name.lower())

  with open(excluded_file) as fin:
    for line in fin.readlines():
      imagenet_excl.add(line.strip())

  return imagenet_excl

imagenet_excl = get_imagenet_excl()
def is_imagenet_tag(word):
  if word in imagenet_excl:
    return False
  if len(word) < 3:
    return False
  if len(wordnet.synsets(word)) == 0:
    return False
  return True

def main():
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
    word_set = word_set.union(word_list)
    print('#%04d %s' % (label_id, text))
    stdout.write('   ')
    for word in word_list:
      stdout.write(' %s' % (word))
    stdout.write('\n\n')

  imagenet_tag_set = set()
  for word in word_set:
    if is_imagenet_tag(word):
      imagenet_tag_set.add(word)
  imagenet_file_t = 'imagenet_tag_set.t'
  utils.save_as_readable(imagenet_tag_set, imagenet_file_t)
  imagenet_file_p = 'imagenet_tag_set.p'
  pickle.dump(imagenet_tag_set, open(imagenet_file_p, 'wb'))

if __name__ == '__main__':
  main()
