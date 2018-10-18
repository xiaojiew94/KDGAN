from utils import data_dir, log_format

import utils

from nltk.corpus import wordnet
from os import path

import argparse
import calendar
import logging
import pickle
import pycountry
import string

logging.basicConfig(level=logging.INFO, format=log_format)

imagenet_file = path.join(data_dir, 'imagenet_tag_set.p')
readable_file = path.join(data_dir, 'wordnet_tag_set.r')
pickle_file = path.join(data_dir, 'wordnet_tag_set.p')
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

  imagenet_tag_set = pickle.load(open(imagenet_file, 'rb'))
  wordnet_excl = wordnet_excl.union(imagenet_tag_set)

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

def main():
  synsets = wordnet.all_synsets('n')
  word_list = [synset.name().split('.')[0] for synset in synsets]

  wordnet_tag_set = set()
  for word in word_list:
    if is_wordnet_tag(word):
      wordnet_tag_set.add(word)
  utils.save_set_readable(wordnet_tag_set, readable_file)
  pickle.dump(wordnet_tag_set, open(pickle_file, 'wb'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--override', action='store_true')
  args = parser.parse_args()
  if not path.isfile(pickle_file) or args.override:
    main()
  else:
    logging.info('do not override')