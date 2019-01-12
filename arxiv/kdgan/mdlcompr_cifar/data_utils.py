from kdgan import config

from os import path
import numpy as np
import tensorflow as tf
import math

class CIFAR():
  def __init__(self, flags):
    tn_input_ts = self.build_input_ts(flags, 'train')
    self.tn_image_ts, self.tn_label_ts = tn_input_ts
    vd_input_ts = self.build_input_ts(flags, 'valid')
    self.vd_image_ts, self.vd_label_ts = vd_input_ts
    self.vd_num_batch = int(math.ceil(flags.valid_size / flags.batch_size))
    self.batch_size = flags.batch_size

  def build_input_ts(self, flags, mode):
    image_size = flags.image_size
    batch_size = flags.batch_size
    num_classes = flags.num_label
    if mode == 'train':
      data_path = flags.train_filepath
    else:
      data_path = flags.valid_filepath

    label_bytes = 1
    label_offset = 0
    depth = 3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes

    data_files = tf.gfile.Glob(data_path)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # read examples from files in the filename queue
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # convert these examples to dense labels and processed images
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    # convert from string to [depth * height * width] to [depth, height, width]
    depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
        [depth, image_size, image_size])
    # convert from [depth, height, width] to [height, width, depth]
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    if mode == 'train':
      image = tf.image.resize_image_with_crop_or_pad(
          image, image_size+4, image_size+4)
      image = tf.random_crop(image, [image_size, image_size, 3])
      image = tf.image.random_flip_left_right(image)
      # brightness/saturation/constrast provides small gains .2%~.5% on cifar
      # image = tf.image.random_brightness(image, max_delta=63. / 255.)
      # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
      image = tf.image.per_image_standardization(image)

      example_queue = tf.RandomShuffleQueue(
          capacity=16 * batch_size,
          min_after_dequeue=8 * batch_size,
          dtypes=[tf.float32, tf.int32],
          shapes=[[image_size, image_size, depth], [1]])
      num_threads = 16
    else:
      image = tf.image.resize_image_with_crop_or_pad(
          image, image_size, image_size)
      image = tf.image.per_image_standardization(image)

      example_queue = tf.FIFOQueue(
          3 * batch_size,
          dtypes=[tf.float32, tf.int32],
          shapes=[[image_size, image_size, depth], [1]])
      num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # read a batch of labels + images from the example queue
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    return images, labels

  def next_batch(self, sess):
    tn_image_np, tn_label_np = sess.run([self.tn_image_ts, self.tn_label_ts])
    return tn_image_np, tn_label_np

  def compute_acc(self, sess, model):
    acc_list = []
    for _ in range(self.vd_num_batch):
      vd_image_np, vd_label_np = sess.run([self.vd_image_ts, self.vd_label_ts])
      feed_dict = {
        model.image_ph:vd_image_np,
        model.hard_label_ph:vd_label_np,
      }
      # predictions = sess.run(model.labels, feed_dict=feed_dict)
      # predictions = np.argmax(predictions, axis=1)
      # groundtruth = np.argmax(vd_label_np, axis=1)
      # acc = 1.0 * np.sum(predictions==groundtruth) / self.batch_size
      acc = sess.run(model.accuracy, feed_dict=feed_dict)
      acc_list.append(acc)
    acc = sum(acc_list) / len(acc_list)
    return acc

if __name__ == '__main__':
  pass






