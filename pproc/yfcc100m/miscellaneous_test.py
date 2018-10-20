from utils import *

import pickle

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

def main():
  in_tag_count = pickle.load(open(in_refined_noun_pfile, 'rb'))
  wn_tag_count = pickle.load(open(wn_refined_noun_pfile, 'rb'))
  user_count = pickle.load(open(refined_user_pfile, 'rb'))

  for tag, count in in_tag_count.items():
    logging.info('%s\t%d' % (tag, count))

if __name__ == '__main__':
  main()