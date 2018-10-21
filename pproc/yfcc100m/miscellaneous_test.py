from utils import *

import pprint
import requests
import json

import logging
logging.basicConfig(level=logging.INFO, format=log_format)

api_key = '906e740ff25b8a697590451f955be478'
api_secret = 'f09f5803c399bace'

def main():
  url_api = 'https://api.flickr.com/services/rest/'
  params = {'api_key': api_key,
            'format': 'json',
            'nojsoncallback': '1',
            'method': 'flickr.people.getInfo',
            'user_id': '85516388@N00',}
  response = requests.get(url_api, params=params)
  info = json.loads(response.text)
  pp = pprint.PrettyPrinter(indent=2)
  pp.pprint(info)

if __name__ == '__main__':
  main()

