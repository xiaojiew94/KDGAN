from kdgan import config
from nets import nets_factory

import tensorflow as tf
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,'')
tf.app.flags.DEFINE_string('trainable_scopes', None, '')
flags = tf.app.flags.FLAGS

def main(_):
    network_fn = nets_factory.get_network_fn(flags.model_name,
            num_classes=config.num_label,
            weight_decay=flags.weight_decay)
    image_size = network_fn.default_image_size

    image_ph = tf.placeholder(tf.float32,
            shape=(None, image_size, image_size, config.channels))
    label_ph = tf.placeholder(tf.float32,
            shape=(None, config.num_label))
        
    logits, end_points = network_fn(image_ph)

    variables_to_restore = []
    for variable in slim.get_model_variables():
        num_params = 1
        for dim in variable.shape:
            num_params *= dim.value
        print('layer {} has {} params'.format(variable.name, num_params))

    for name, tensor in end_points.items():
        print('{} {}'.format(name, tensor.shape))

if __name__ == '__main__':
    tf.app.run()