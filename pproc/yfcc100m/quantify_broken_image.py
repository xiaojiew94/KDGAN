from nltk.corpus import wordnet
from operator import itemgetter
from sys import stdout

import string

dataset_file = '/data/yfcc100m/yfcc100m_dataset'
tag_file = '/data/yfcc100m/yfcc100m_tag'
tmp_file = '/data/yfcc100m/yfcc100m_tmp'

sup_tag = 20

num_field = 25
idx_field = 10
idx_marker = 24

sep_tag = ','
sep_word = '+'


from nltk.corpus import wordnet

import string
import calendar
import pycountry

def get_spatial():
  spatial_noun = set()
  for country in pycountry.countries:
    if hasattr(country, 'alpha_2'):
      spatial_noun.add(country.alpha_2.lower())
    if hasattr(country, 'alpha_3'):
      spatial_noun.add(country.alpha_3.lower())
    if hasattr(country, 'name'):
      spatial_noun.add(country.name.lower())
    if hasattr(country, 'official_name'):
      spatial_noun.add(country.official_name.lower())
  for country in pycountry.historic_countries:
    if hasattr(country, 'alpha_3'):
      spatial_noun.add(country.alpha_3.lower())
    if hasattr(country, 'alpha_4'):
      spatial_noun.add(country.alpha_4.lower())
    if hasattr(country, 'name'):
      spatial_noun.add(country.name.lower())
  for subdivision in pycountry.subdivisions:
    if hasattr(subdivision, 'name'):
      spatial_noun.add(subdivision.name.lower())
    if hasattr(subdivision, 'code'):
      spatial_noun.add(subdivision.code.lower())
  return spatial_noun

def get_temporal():
  temporal_noun = set()
  for month in range(1, 13):
    temporal_noun.add(calendar.month_name[month].lower())
    temporal_noun.add(calendar.month_abbr[month].lower())
  for day in range(0, 7):
    temporal_noun.add(calendar.day_name[day].lower())
    temporal_noun.add(calendar.day_abbr[day].lower())
  return temporal_noun

v_char = string.ascii_lowercase + '-'
spatial_noun = get_spatial()
temporal_noun = get_temporal()
def is_valid(noun):
  if any(c not in v_char for c in noun):
    return False
  if noun in spatial_noun:
    return False
  if noun in temporal_noun:
    return False
  synsets = wordnet.synsets(noun)
  for synset in synsets:
    if synset.name().split('.')[1] != 'n':
      return False
  return True

def main():
  unamb_nouns = set()
  wn_nouns = {synset.name().split('.')[0] for synset in wordnet.all_synsets('n')}
  for noun in wn_nouns:
    # print('%s:%s' % (noun, is_valid(noun)))
    if is_valid(noun):
      unamb_nouns.add(noun)
  print('%d->%d' % (len(wn_nouns), len(unamb_nouns)))

  num_tag = 0
  tags = []
  with open(tag_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split()
      tag, count = fields[0], int(fields[1])

      valid = tag in unamb_nouns
      if valid:
        num_tag += 1
        tags.append(tag)
  assert num_tag == len(tags)
  with open(tmp_file, 'w') as fout:
    for tag in tags:
      fout.write('%s\n' % (tag))
  print('%d valid tags' % (num_tag))

def test():
  tag_count = {}
  t_line = 0
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      assert len(fields) == num_field

      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_field]) == 0: # no tags
        continue
      tags = fields[idx_field].split(sep_tag)
      for tag in tags:
        tag = tag.lower()
        words = tag.split(sep_word)
        valid = True
        for word in words:
          if not is_valid(word):
            valid = False
            break
        if valid:
          tag_count[tag] = tag_count.get(tag, 0) + 1

      t_line += 1
      # if t_line == 50000:
      #   break
      if (t_line % 5000000) == 0:
        print('line#%09d' % (t_line))
  print('%s contains %d lines in total' % (dataset_file, t_line))

  tag_count = sorted(tag_count.items(), key=itemgetter(1), reverse=True)
  with open(tag_file, 'w') as fout:
    for tag, count in tag_count:
      if count < sup_tag:
        break
      fout.write('%s\t%d\n' % (tag, count))

if __name__ == '__main__':
  main()


