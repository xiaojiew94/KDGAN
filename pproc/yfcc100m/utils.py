data_dir = 'data'
log_format = '%(pathname)-25s%(message)s'

def save_set_readable(item_set, outfile):
  with open(outfile, 'w') as fout:
    for item in sorted(item_set):
      fout.write('%s\n' % (item))
