data_dir = 'data'
log_format = '%(pathname)-25s%(message)s'

num_field = 25
idx_user = 3
idx_tag = 10
idx_marker = 24
sep_field = '\t'
sep_tag = ','

def save_set_readable(item_set, outfile):
  with open(outfile, 'w') as fout:
    for item in sorted(item_set):
      fout.write('%s\n' % (item))
