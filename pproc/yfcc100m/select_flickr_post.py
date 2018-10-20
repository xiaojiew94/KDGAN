from utils import *

import argparse
import pickle

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

def main():
  in_tag_count = pickle.load(open(in_refined_noun_pfile, 'rb'))
  wn_tag_count = pickle.load(open(wn_refined_noun_pfile, 'rb'))
  user_count = pickle.load(open(refined_user_pfile, 'rb'))

  num_tag = len(wn_tag_count) + len(in_tag_count)
  num_user = len(user_count)
  logging.info('#tag=%d #user=%d' % (num_tag, num_user))

  num_post = 0
  tot_line = 0
  with open(yfcc_dataset_file) as fin, \
       open(yfcc_sample_file, 'w') as fout:
    while True:
      line = fin.readline()
      if not line:
        break
      tot_line += 1
      if (tot_line % 20000000) == 0:
        break
        logging.info('#line=%09d #post=%d' % (tot_line, num_post))

      fields = line.strip().split(sep_field)
      assert len(fields) == num_field
      if fields[idx_marker] != '0': # not image
        continue
      if len(fields[idx_tag]) == 0: # no tags
        continue
      user = fields[idx_user]
      if user not in user_count:
        continue
      in_tags, wn_tags = [], []
      for tag in fields[idx_tag].split(sep_tag):
        if tag in in_tag_count:
          in_tags.append(tag)
        if tag in wn_tag_count:
          wn_tags.append(tag)
      if len(in_tags) + len(wn_tags) == 0:
        continue
      title = fields[idx_title]
      description = fields[idx_description]
      is_valid = True
      for tag in in_tags + wn_tags:
        if tag not in title and tag not in description:
          is_valid = False
          break
      if not is_valid:
        continue
      num_post += 1
      fout.write(line)
  logging.info('#line=%09d #post=%d' % (tot_line, num_post))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--override', action='store_true')
  args = parser.parse_args()
  if not path.isfile(yfcc_sample_file) or args.override:
    main()
  else:
    logging.info('do not override')