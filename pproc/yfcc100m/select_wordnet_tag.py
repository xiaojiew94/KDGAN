import utils

from nltk.corpus import wordnet

import string
import calendar
import pickle
import pycountry

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

  imagenet_file_p = 'imagenet_tag_set.p'
  imagenet_tag_set = pickle.load(open(imagenet_file_p, 'rb'))
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
  wordnet_file_t = 'wordnet_tag_set.t'
  utils.save_as_readable(wordnet_tag_set, wordnet_file_t)
  wordnet_file_p = 'wordnet_tag_set.p'
  pickle.dump(wordnet_tag_set, open(wordnet_file_p, 'wb'))

if __name__ == '__main__':
  main()
