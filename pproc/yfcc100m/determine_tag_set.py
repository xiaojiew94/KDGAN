from nltk.corpus import wordnet
from urllib import request

import string
import calendar
import pickle
import pycountry
import re

def save_as_readable(tag_set, file_t):
  with open(file_t, 'w') as fout:
    for tag in sorted(tag_set):
      fout.write('%s\n' % (tag))

worldcity_file = '/data/yfcc100m/worldcitiespop.txt'
def get_wordnet_excl():
  wordnet_excl = set()

  for month in range(1, 13):
    wordnet_excl.add(calendar.month_name[month].lower())
    wordnet_excl.add(calendar.month_abbr[month].lower())
  for day in range(0, 7):
    wordnet_excl.add(calendar.day_name[day].lower())
    wordnet_excl.add(calendar.day_abbr[day].lower())

  for country in pycountry.countries:
    wordnet_excl.add(country.alpha_2.lower())
    wordnet_excl.add(country.alpha_3.lower())
    wordnet_excl.add(country.name.lower())
  for country in pycountry.historic_countries:
    wordnet_excl.add(country.alpha_3.lower())
    wordnet_excl.add(country.alpha_4.lower())
    wordnet_excl.add(country.name.lower())
  for subdivision in pycountry.subdivisions:
    wordnet_excl.add(subdivision.name.lower())
    wordnet_excl.add(subdivision.code.lower())

  with open(worldcity_file, encoding='iso-8859-1') as fin:
    line = fin.readline()
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split(',')
      country_code, city = fields[0], fields[1]
      if ' ' in city:
        continue
      wordnet_excl.add(city)
      wordnet_excl.add(country_code)

  return wordnet_excl

wordnet_char = string.ascii_lowercase + '-'
wordnet_excl = get_wordnet_excl()
def is_wordnet_tag(word):
  if word in wordnet_excl:
    return False
  if len(word) < 4:
    return False
  if any(c not in wordnet_char for c in word):
    return False
  synsets = wordnet.synsets(word)
  for synset in synsets:
    if synset.name().split('.')[1] != 'n':
      return False
  return True

def get_imagenet_excl():
  imagenet_excl = set()

  for country in pycountry.countries:
    imagenet_excl.add(country.name.lower())
  for country in pycountry.historic_countries:
    imagenet_excl.add(country.name.lower())
  for subdivision in pycountry.subdivisions:
    imagenet_excl.add(subdivision.name.lower())

  with open('imagenet_exclude.t') as fin:
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

def save_wordnet_tag_set():
  synsets = wordnet.all_synsets('n')
  word_list = [synset.name().split('.')[0] for synset in synsets]

  wordnet_tag_set = set()
  for word in word_list:
    if is_wordnet_tag(word):
      wordnet_tag_set.add(word)
  wordnet_file_t = 'wordnet_tag_set.t'
  save_as_readable(wordnet_tag_set, wordnet_file_t)
  wordnet_file_p = 'wordnet_tag_set.p'
  pickle.dump(wordnet_tag_set, open(wordnet_file_p, 'wb'))

def save_imagenet_tag_set():
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

  imagenet_tag_set = set()
  for word in word_set:
    if is_imagenet_tag(word):
      imagenet_tag_set.add(word)
  imagenet_file_t = 'imagenet_tag_set.t'
  save_as_readable(imagenet_tag_set, imagenet_file_t)
  imagenet_file_p = 'imagenet_tag_set.p'
  pickle.dump(imagenet_tag_set, open(imagenet_file_p, 'wb'))

def main():
  save_wordnet_tag_set()
  save_imagenet_tag_set()

if __name__ == '__main__':
  main()
