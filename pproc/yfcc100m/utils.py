
def save_as_readable(tag_set, file_t):
  with open(file_t, 'w') as fout:
    for tag in sorted(tag_set):
      fout.write('%s\n' % (tag))
