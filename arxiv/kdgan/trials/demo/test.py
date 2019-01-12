# import tensorflow as tf
# from os import path
# temp_dir = 'temp'
# alp_ckpt = path.join(temp_dir, 'alp.ckpt')
# bet_ckpt = path.join(temp_dir, 'bet.ckpt')
# def save_var():
#     alp = tf.get_variable('alp', shape=[3], initializer = tf.zeros_initializer)
#     bet = tf.get_variable('bet', shape=[5], initializer = tf.zeros_initializer)
#     inc_alp = alp.assign(alp + 1)
#     dec_bet = bet.assign(bet - 1)
#     init_op = tf.global_variables_initializer()
#     saver_alp = tf.train.Saver({'alp':alp})
#     saver_bet = tf.train.Saver({'bet':bet})
#     with tf.Session() as sess:
#         sess.run(init_op)
#         inc_alp.op.run()
#         dec_bet.op.run()
#         path_alp = saver_alp.save(sess, alp_ckpt)
#         print('save alp in %s' % path_alp)
#         path_bet = saver_bet.save(sess, bet_ckpt)
#         print('save bet in %s' % path_bet)
# def restore_var():
#     alp = tf.get_variable('alp', shape=[3])
#     # bet = tf.get_variable('bet', shape=[5])
#     bet = tf.get_variable('bet', shape=[5], initializer = tf.zeros_initializer)
#     saver_alp = tf.train.Saver({'alp':alp})
#     # saver_bet = tf.train.Saver({'bet':bet})
#     with tf.Session() as sess:
#         # saver_alp.restore(sess, alp_ckpt)
#         # saver_bet.restore(sess, bet_ckpt)
#         saver_alp.restore(sess, alp_ckpt)
#         bet.initializer.run()
#         print('alp: %s' % alp.eval())
#         print('bet: %s' % bet.eval())
# def main():
#     # save_var()
#     restore_var()
# if __name__ == '__main__':
#     main()


# [0, 1, 2, 3, 4 ,...]
# x = tf.range(1, 10, name="x")
# x = ['0', '1', '2', '3', '4', '5']
# # A queue that outputs 0,1,2,3,...
# range_q = tf.train.range_input_producer(limit=5, shuffle=False)
# slice_end = range_q.dequeue()
# # Slice x to variable length, i.e. [0], [0, 1], [0, 1, 2], ....
# y = tf.slice(x, [0], [slice_end], name="y")
# batched_data = tf.train.batch(
#     tensors=[y],
#     batch_size=5,
#     dynamic_pad=True,
#     name="y_batch"
# )
# # Run the graph
# # tf.contrib.learn takes care of starting the queues for us
# res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)
# # Print the result
# print("Batch shape: {}".format(res[0]["y"].shape))
# print(res[0]["y"])


# labels = [
# 	[0.0, 0.5, 0.0, 0.5],
# 	[0.0, 0.5, 0.0, 0.5],
# 	[0.0, 0.5, 0.0, 0.5],
# 	[0.0, 0.5, 0.0, 0.5],
# ]
# logits = [
# 	[0.1, 0.2, 0.3, 0.4],
# 	[0.2, 0.3, 0.4, 0.1],
# 	[0.3, 0.4, 0.1, 0.2],
# 	[0.4, 0.1, 0.2, 0.3],
# ]
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
# loss = tf.losses.sigmoid_cross_entropy(labels, logits)
# with tf.Session() as sess:
# 	res, = sess.run([loss])
# 	print(res)

# import tensorflow as tf

# def main():
  
#   starter_learning_rate = 0.1
#   end_learning_rate = 0.01
#   decay_steps = 10
#   learning_rate_ts = tf.train.polynomial_decay(
#       starter_learning_rate, global_step_ts, decay_steps, end_learning_rate,
#       power=0.5)
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate_ts)
#   loss_ts = tf.Variable(0.0)
#   train_op = optimizer.minimize(loss_ts, global_step=global_step_ts)
#   num_batch = 16
#   init_op = tf.global_variables_initializer()
#   with tf.Session() as sess:
#     sess.run(init_op)
#     for batch in range(num_batch):
#       result = sess.run([global_step_ts, learning_rate_ts])
#       global_step, learning_rate = result
#       print('batch %02d: %d %f' % (batch, global_step, learning_rate))
#       sess.run(train_op)

# if __name__ == '__main__':
# 	main()

import numpy as np
import tensorflow as tf

image_np = [
  [[[1], [2]], [[3], [4]]],
  [[[3], [4]], [[5], [6]]],
  [[[5], [6]], [[7], [8]]],
  [[[7], [8]], [[9], [0]]],
]
image_np = [
  [[[6], [1]], [[3], [0]]],
  [[[2], [8]], [[6], [5]]],
  [[[9], [7]], [[7], [8]]],
  [[[5], [3]], [[5], [1]]],
]
image_np = np.asarray(image_np, dtype=np.float32)
print(image_np.shape)

num_feature = image_np.shape[1]*image_np.shape[2]*image_np.shape[3]
image_ts = tf.reshape(image_np, [-1, num_feature])
with tf.Session() as sess:
  print(sess.run(image_ts))







