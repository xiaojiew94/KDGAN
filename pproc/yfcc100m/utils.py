from os import path

yfcc_dataset_file = '/data/yfcc100m/yfcc100m_dataset'
yfcc_sample_file = '/data/yfcc100m/yfcc100m_sample'
flickr_image_dir = '/data/yfcc100m/flickr_images'

data_dir = 'data'
in_all_noun_rfile = path.join(data_dir, 'in_all_noun.r')
in_all_noun_pfile = path.join(data_dir, 'in_all_noun.p')
wn_all_noun_rfile = path.join(data_dir, 'wn_all_noun.r')
wn_all_noun_pfile = path.join(data_dir, 'wn_all_noun.p')
in_initial_noun_pfile = path.join(data_dir, 'in_initial_noun.p')
wn_initial_noun_pfile = path.join(data_dir, 'wn_initial_noun.p')
initial_user_pfile = path.join(data_dir, 'initial_user.p')
in_refined_noun_pfile = path.join(data_dir, 'in_refined_noun.p')
wn_refined_noun_pfile = path.join(data_dir, 'wn_refined_noun.p')
refined_user_pfile = path.join(data_dir, 'refined_user.p')

min_user = 20 # 40
min_in_tag = 20 # 40
min_wn_tag = 20 # 200

num_field = 25
idx_user = 3
idx_title = 8
idx_description = 9
idx_tag = 10
idx_image_url = 16
idx_marker = 24
sep_field = '\t'
sep_tag = ','

log_format = '%(pathname)-25s%(message)s'

def save_set_readable(item_set, outfile):
  with open(outfile, 'w') as fout:
    for item in sorted(item_set):
      fout.write('%s\n' % (item))

def get_image_file(image_fold_dir, image_url):
  image_url = image_url.replace('http://', '')
  image_file = path.join(image_fold_dir, *image_url.split('/'))
  return image_file







