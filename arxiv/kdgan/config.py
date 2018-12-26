import sys
from os import path

config_file = path.realpath(__file__)
kdgan_dir = path.dirname(config_file)
pypkg_dir = path.dirname(kdgan_dir)
# print('config kdgan_dir:%s' % (kdgan_dir))
# print('config pypkg_dir:%s' % (pypkg_dir))
logs_dir = path.join(kdgan_dir, 'logs')
temp_dir = path.join(kdgan_dir, 'temp')
ckpt_dir = path.join(kdgan_dir, 'checkpoints')
pickle_dir = path.join(kdgan_dir, 'pickles')
picture_dir = path.join(kdgan_dir, 'pltfigure/pictures')

home_dir = path.expanduser('~')
proj_dir = path.join(home_dir, 'Projects')
data_dir = path.join(proj_dir, 'data')
yfcc_dir = path.join(data_dir, 'yfcc100m')
surv_dir = path.join(yfcc_dir, 'survey_data')

yfcc_top_dir = path.join(yfcc_dir, 'yfcc_top')
yfcc_rnd_dir = path.join(yfcc_dir, 'yfcc_rnd')

image_dir = path.join(yfcc_dir, 'images')
mnist_dir = path.join(data_dir, 'mnist')
cifar_dir = path.join(data_dir, 'cifar')
cifar_ext = path.join(cifar_dir, 'cifar-10-batches-bin')

slim_dir = path.join(pypkg_dir, 'slim')
sys.path.insert(0, slim_dir)

rawtag_file = path.join(yfcc_dir, 'sample_00')
sample_file = path.join(yfcc_dir, 'sample_09')
tfrecord_tmpl = '{0}_{1}_{2:03d}.{3}.tfrecord'

user_key = 'user'
image_key = 'image'
text_key = 'text'
label_key = 'label'
file_key = 'file'
unk_token = 'unk'
pad_token = ' '

# channels = 3
# num_label = 100
num_readers = 4
num_threads = 4
num_preprocessing_threads = 4
train_batch_size = 32
valid_batch_size = 100

max_norm = 10







