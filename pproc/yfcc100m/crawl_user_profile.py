from utils import *

from os import path

import json
import os
import pickle
import requests
import time

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

url_api = 'https://api.flickr.com/services/rest/'
api_key = '906e740ff25b8a697590451f955be478'
api_secret = 'f09f5803c399bace'

def crawl(user):
  user_file = path.join(user_profile_dir, '%s.json' % (user))
  if path.isfile(user_file):
    return

  params = {'api_key': api_key,
            'format': 'json',
            'nojsoncallback': '1',
            'method': 'flickr.people.getInfo',
            'user_id': user,}
  response = requests.get(url_api, params=params)
  info = json.loads(response.text)
  with open(user_file, 'w') as fout:
      json.dump(info, fout)

def main():
  if not path.exists(user_profile_dir):
    os.makedirs(user_profile_dir)

  num_user = 0
  user_count = pickle.load(open(refined_user_pfile, 'rb'))
  for user in user_count.keys():
    crawl(user)
    num_user += 1
    if (num_user % 100) == 0:
      logging.info('#user=%d' % (num_user))
    time.sleep(1)

if __name__ == '__main__':
  main()
