import tensorflow as tf

import mock

variables = []
constants = []

def custom_getter(getter, name, **kwargs):
  trainable = kwargs['trainable']
  kwargs['trainable'] = False
  variable = getter(name, **kwargs)
  if trainable:
    variables.append(variable)
  else:
    constants.append(variable)
  return variable

# print(len(variables), len(constants))

batch_size = 32
num_dims = 10
dtype = tf.float64
stddev = 0.01

def func():
  x = tf.get_variable(
      "x",
      shape=[batch_size, num_dims],
      dtype=dtype,
      initializer=tf.random_normal_initializer(stddev=stddev))

  w = tf.get_variable("w",
                      shape=[batch_size, num_dims, num_dims],
                      dtype=dtype,
                      initializer=tf.random_uniform_initializer(),
                      trainable=False)
  y = tf.get_variable("y",
                      shape=[batch_size, num_dims],
                      dtype=dtype,
                      initializer=tf.random_uniform_initializer(),
                      trainable=False)

  product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
  loss = tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))
  return loss
original_get_variable = tf.get_variable
def custom_get_variable(*args, **kwargs):
  if hasattr(kwargs, "custom_getter"):
    raise AttributeError("Custom getters are not supported for optimizee "
                         "variables.")
  return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

with mock.patch("tensorflow.get_variable", custom_get_variable):
  loss = func()

# print(len(variables), len(constants))
gradients = tf.gradients(loss, variables)
# print(len(variables), len(gradients))

for gradient, variable in zip(gradients, variables):
  print(gradient.shape, variable.shape)




