def init_tag_count(file_p):
  tag_set = pickle.load(open(file_p, 'rb'))
  tag_count = {tag:0 for tag in tag_set}
  return tag_count

  n_post = 0
  t_line = 0
  with open(dataset_file) as fin:
    while True:
      line = fin.readline()
      if not line:
        break

      fields = line.strip().split('\t')
      assert len(fields) == num_field

      if fields[idx_marker] != '0': # not image
        continue

      t_line += 1
      is_valid = False
      tags = fields[idx_field].split(sep_tag)
      for tag in tags:
        tag = tag.lower()
        if (tag in imagenet_tag_count or
            tag in wordnet_tag_count):
          is_valid = True
          break
      if is_valid:
        n_post += 1

      # if t_line == 50000:
      #   break
      if (t_line % 5000000) == 0:
        print('line#%09d #post=%d' % (t_line, n_post))
  print('%s contains %d lines in total' % (dataset_file, t_line))
