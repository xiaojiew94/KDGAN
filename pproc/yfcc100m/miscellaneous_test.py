from operator import itemgetter
from urllib import request


def main():
  base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/'
  synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
  synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

  filename, _ = request.urlretrieve(synset_url)
  synset_list = [s.strip() for s in open(filename).readlines()]
  num_synsets_in_ilsvrc = len(synset_list)
  assert num_synsets_in_ilsvrc == 1000

  filename, _ = request.urlretrieve(synset_to_human_url)
  synset_to_human_list = open(filename).readlines()
  num_synsets_in_all_imagenet = len(synset_to_human_list)
  assert num_synsets_in_all_imagenet == 21842

  synset_to_human = {}
  for s in synset_to_human_list:
    parts = s.strip().split('\t')
    assert len(parts) == 2
    synset = parts[0]
    human = parts[1]
    synset_to_human[synset] = human

  label_index = 1
  labels_to_names = {0: 'background'}
  for synset in synset_list:
    name = synset_to_human[synset]
    labels_to_names[label_index] = name
    label_index += 1

  print(labels_to_names)

def test():
  file = '/home/xiaojie/Projects/data/yfcc100m/yfcc_rnd/yfcc_rnd.data'
  # file = '/home/xiaojie/Projects/data/yfcc100m/yfcc_top/yfcc10k.data'
  # print('miscellaneous test')
  tag_count = {}
  with open(file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      tags = fields[-1].split()
      for tag in tags:
        tag_count[tag] = tag_count.get(tag, 0) + 1

  tag_count = sorted(tag_count.items(), key=itemgetter(1), reverse=True)
  for tag, count in tag_count:
    print('%s\t%d' % (tag, count))

if __name__ == '__main__':
  main()