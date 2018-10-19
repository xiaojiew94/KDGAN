import pprint
import requests
import json

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
in_tag_count_file = path.join(data_dir, 'imagenet_tag_count.p')
wn_tag_count_file = path.join(data_dir, 'wordnet_tag_count.p')
user_count_file = path.join(data_dir, 'flickr_user_count.p')
  in_tag_count = pickle.load(open(in_tag_count_file, 'rb'))
  wn_tag_count = pickle.load(open(wn_tag_count_file, 'rb'))
  user_count = pickle.load(open(user_count_file, 'rb'))
   
if __name__ == '__main__':
  main()