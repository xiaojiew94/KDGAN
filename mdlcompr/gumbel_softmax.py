import tensorflow as tf
import numpy as np

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax(y / temperature)

logits = np.random.rand(3, 4)
gumbel_logits = gumbel_softmax(logits, 1.0)
with tf.Session() as tf:
  for i in range(100):
    logits = tf.run(gumbel_logits)
    if (i + 1) % 20 == 0:
      print(i + 1)
    if np.isnan(logits).any():
      print('logits has nan')