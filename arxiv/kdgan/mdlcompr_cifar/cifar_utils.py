from kdgan import config
from flags import flags

from os import path
import numpy as np
import tensorflow as tf
import math

def main(_):
  infile = path.join(config.cifar_ext, 'data_batch_1.bin')
  exp_image = flags.train_size
  num_label = 10
  exp_label = int(exp_image / num_label)

  num_image = 0
  
  label_bytes = 1
  height = 32
  width = 32
  channel = 3
  image_bytes = height * width * channel
  record_bytes = label_bytes + image_bytes

  record_list = []
  label_count = {}

  with open(infile, 'rb') as fin:
    while True:
      record_bt = fin.read(record_bytes)
      if not record_bt:
        break
      record_np = np.fromstring(record_bt, dtype=np.uint8)
      label = record_np[0]
      count = label_count.get(label, 0)
      if count == exp_label:
        continue
      record_list.append(record_np)
      label_count[label] = count + 1
      num_image += 1

  print('#image=%d' % (num_image))

  # for label, count in label_count.items():
  #   print('label=%d count=%d' % (label, count))

  outdir = path.dirname(infile)
  outfile = path.join(outdir, 'cifar10_%d.bin' % (exp_image))
  with open(outfile, 'wb') as fout:
    for record_np in record_list:
      record_bt = record_np.tobytes()
      fout.write(record_bt)

if __name__ == '__main__':
  tf.app.run()








