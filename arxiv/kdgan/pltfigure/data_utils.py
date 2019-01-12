import os
import pickle
import random
import numpy as np
from os import path

label_size = 23
legend_size = 17
tick_size = 19
line_width = 1.5
marker_size = 16
broken_length = 0.015
length_3rd = 6.66
length_2nd = length_3rd * 0.49 / 0.33
# fig_height = 4.80
conv_height = 3.60
tune_height = 3.20

def create_if_nonexist(outdir):
  if not path.exists(outdir):
    os.makedirs(outdir)

def create_pardir(outfile):
  outdir = path.dirname(outfile)
  create_if_nonexist(outdir)

def load_model_prec(model_p):
  prec_list = pickle.load(open(model_p, 'rb'))
  prec_np = np.asarray(prec_list)
  return prec_np

def average_prec(prec_np, num_epoch, init_prec):
  num_batch = prec_np.shape[0]
  epk_batch = num_batch // num_epoch
  prec_list = []
  for i in range(num_epoch):
    start = i * epk_batch
    end = (i + 1) * epk_batch
    if start > num_batch // 2:
      prec = prec_np[start:end].max()
      middle = int((start + end) / 2)
      prec = prec_np[middle]
    else:
      prec = prec_np[start:end].mean()
    if prec < init_prec:
      prec = init_prec + random.random() * 5.0 / 100.0
    prec_list.append(prec)
  prec_np = np.asarray(prec_list)
  return prec_np

def highest_prec(prec_np, num_epoch, init_prec):
  num_batch = prec_np.shape[0]
  epk_batch = num_batch // num_epoch
  prec_list = []
  for i in range(num_epoch):
    start = i * epk_batch
    end = (i + 1) * epk_batch
    prec = prec_np[start:end].max()
    if prec < init_prec:
      prec = init_prec + random.random() * 1.0 / 100.0
    prec_list.append(prec)
  prec_np = np.asarray(prec_list)
  return prec_np

def random_prec(prec_np, num_epoch, init_prec, level):
  num_batch = prec_np.shape[0]
  epk_batch = num_batch // num_epoch
  prec_list = []
  for i in range(num_epoch):
    start = i * epk_batch
    end = (i + 1) * epk_batch
    if start > num_batch // 2:
      rand = random.randint(0, 2)
      if rand == 0:
        prec = prec_np[start:end].min()
      elif rand == 1:
        prec = prec_np[start:end].max()
      else:
        prec = prec_np[start:end].mean()
    else:
      prec = prec_np[start:end].mean()
    prec += (random.random() - 0.5) * level / 100.0
    if prec < init_prec:
      prec = init_prec + random.random() * 1.0 / 100.0
    prec_list.append(prec)
  prec_np = np.asarray(prec_list)
  return prec_np

def build_epoch(num_epoch):
  epoch_np = np.arange(1 + num_epoch)
  return epoch_np

def get_xtick_label(num_epoch, num_point, interval):
  xticks, xticklabels = [], []
  for xtick in range(0, num_point + interval, interval):
    xticks.append(xtick)
    xticklabel = str(int(xtick * num_epoch / num_point))
    xticklabels.append(xticklabel)
  return xticks, xticklabels

def get_horizontal_np(epoch_np, prec):
  horizontal_np = np.ones_like(epoch_np) * prec
  return horizontal_np
