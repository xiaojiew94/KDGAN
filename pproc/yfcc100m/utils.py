from os import path

data_dir = 'data'
log_format = '%(pathname)-25s%(message)s'

dataset_file = '/data/yfcc100m/yfcc100m_dataset'

in_all_noun_rfile = path.join(data_dir, 'in_all_noun.r')
in_all_noun_pfile = path.join(data_dir, 'in_all_noun.p')

wn_all_noun_rfile = path.join(data_dir, 'wn_all_noun.r')
wn_all_noun_pfile = path.join(data_dir, 'wn_all_noun.p')

min_user = 20 # 40
min_in_tag = 20 # 40
min_wn_tag = 20 # 200

in_initial_noun_pfile = path.join(data_dir, 'in_initial_noun.p')
wn_initial_noun_pfile = path.join(data_dir, 'wn_initial_noun.p')
initial_user_pfile = path.join(data_dir, 'initial_user.p')

in_refined_noun_pfile = path.join(data_dir, 'in_refined_noun.p')
wn_refined_noun_pfile = path.join(data_dir, 'wn_refined_noun.p')
refined_user_pfile = path.join(data_dir, 'refined_user.p')

num_field = 25
idx_user = 3
idx_title = 8
idx_description = 9
idx_tag = 10
idx_marker = 24
sep_field = '\t'
sep_tag = ','

def save_set_readable(item_set, outfile):
  with open(outfile, 'w') as fout:
    for item in sorted(item_set):
      fout.write('%s\n' % (item))
