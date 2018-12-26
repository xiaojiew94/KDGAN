import math
import numpy as np

################################################################
#
# jingwei metric
#
################################################################

VALID_LABEL_SET_AP = set([-1, 0, 1, 2, 3])
VALID_LABEL_SET_NDCG2 = set([0, 1, 2, 3])

class MetricScorer:
  def __init__(self, k=0):
    self.k = k

  def score(self, sorted_labels):
    return 0.0

  def getLength(self, sorted_labels):
    length = self.k
    if length>len(sorted_labels) or length<=0:
      length = len(sorted_labels)
    return length    

  def name(self):
    if self.k > 0:
      return "%s@%d" % (self.__class__.__name__.replace("Scorer",""), self.k)
    return self.__class__.__name__.replace("Scorer","")

class APScorer (MetricScorer):
  def __init__(self, k):
    MetricScorer.__init__(self, k)

  def score(self, sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
      return 0.0
        
    length = self.getLength(sorted_labels)
    ap = 0.0
    rel = 0
    
    for i in range(length):
      lab = sorted_labels[i]
      assert(lab in VALID_LABEL_SET_AP)
      if lab >= 1:
        rel += 1
        ap += float(rel) / (i+1.0)
    ap /= nr_relevant
    return ap

class RRScorer (MetricScorer):
  def score(self, sorted_labels):
    for i in range(len(sorted_labels)):
      if 1 <= sorted_labels[i]:
        return 1.0/(i+1)
    return 0.0

class PrecisionScorer (MetricScorer):
  def score(self, sorted_labels):
    length = self.getLength(sorted_labels)

    rel = 0
    for i in range(length):
      if sorted_labels[i] >= 1:
        rel += 1

    return float(rel)/length

class NDCGScorer (PrecisionScorer):
  def score(self, sorted_labels):
    if not sorted_labels:
      return 0.0
    d = self.getDCG(sorted_labels)
    d2 = self.getIdealDCG(sorted_labels) 
    # print('\n', d, d2)
    return d/d2

  def getDCG(self, sorted_labels):
    length = self.getLength(sorted_labels)

    dcg = max(sorted_labels[0], 0)
    # print(dcg)
    for i in range(1, length):
      rel = max(sorted_labels[i], 0)
      dcg += float(rel)/math.log(i+1, 2)
      # print(i, sorted_labels[i], math.log(i+1,2), float(sorted_labels[i])/math.log(i+1, 2))
    return dcg

  def getIdealDCG(self, sorted_labels):
    ideal_labels = sorted(sorted_labels, reverse=True)
    assert(ideal_labels[0] > 0), len(ideal_labels)
    return self.getDCG(ideal_labels)

class NDCG2Scorer (NDCGScorer):
  def getDCG(self, sorted_labels):
    length = self.getLength(sorted_labels)
    dcg = 0
    for i in range(0, length):
      rel_i = max(sorted_labels[i], 0) 
      # assert(rel_i in VALID_LABEL_SET_NDCG2)
      dcg += (math.pow(2,rel_i) - 1) / math.log(i+2, 2)
    return dcg

class RecallScorer (MetricScorer):
  def getLength(self, sorted_labels):
    length = 0
    for i in range(len(sorted_labels)):
      if 1 <= sorted_labels[i]:
        length += 1
    # print('{}\nlength={}'.format(sorted_labels, length))
    return length

  def score(self, sorted_labels):
    length = self.getLength(sorted_labels)

    rel = 0
    for i in range(length):
      if sorted_labels[i] >= 1:
        rel += 1

    return float(rel)/length

def getScorer(name):
  mapping = {
    "P":PrecisionScorer,
    "AP":APScorer,
    "RR":RRScorer,
    "NDCG":NDCGScorer,
    "NDCG2":NDCG2Scorer,
    "R":RecallScorer,
  }
  elems = name.split("@")
  if len(elems) == 2:
    k = int(elems[1])
  else:
    k = 0
  return mapping[elems[0]](k)

def eval_tagrecom(logits, labels, cutoff):
  p3_scorer = getScorer('P@3')
  p5_scorer = getScorer('P@5')
  r3_scorer = getScorer('R@3')
  r5_scorer = getScorer('R@5')
  ndcg3_scorer = getScorer('NDCG2@3')
  ndcg5_scorer = getScorer('NDCG2@5')
  ap_scorer = getScorer('AP')
  rr_scorer = getScorer('RR')

  predictions = np.argsort(-logits, axis=1)
  batch_size, _ = labels.shape
  p3, p5, f3, f5, ndcg3, ndcg5, ap, rr = [], [], [], [], [], [], [], []
  for batch in range(batch_size):
    label_bt = labels[batch, :]
    label_bt = np.nonzero(label_bt)[0]
    prediction_bt = predictions[batch, :]
    sorted_label_bt = [int(p in label_bt) for p in prediction_bt]
    p3_bt = p3_scorer.score(sorted_label_bt)
    p5_bt = p5_scorer.score(sorted_label_bt)
    r3_bt = r3_scorer.score(sorted_label_bt)
    r5_bt = r5_scorer.score(sorted_label_bt)
    f3_bt, f5_bt = 0.0, 0.0
    if (p3_bt + r3_bt) != 0.0:
      f3_bt = 2 * p3_bt * r3_bt / (p3_bt + r3_bt)
    if (p5_bt + r5_bt) != 0.0:
      f5_bt = 2 * p5_bt * r5_bt / (p5_bt + r5_bt)
    p3.append(p3_bt)
    p5.append(p5_bt)
    f3.append(f3_bt)
    f5.append(f5_bt)
    ndcg3.append(ndcg3_scorer.score(sorted_label_bt))
    ndcg5.append(ndcg5_scorer.score(sorted_label_bt))
    ap.append(ap_scorer.score(sorted_label_bt))
    rr.append(rr_scorer.score(sorted_label_bt))
  p3 = float(np.mean(p3))
  p5 = float(np.mean(p5))
  f3 = float(np.mean(f3))
  f5 = float(np.mean(f5))
  ndcg3 = float(np.mean(ndcg3))
  ndcg5 = float(np.mean(ndcg5))
  ap = float(np.mean(ap))
  rr = float(np.mean(rr))
  return p3, p5, f3, f5, ndcg3, ndcg5, ap, rr

################################################################
#
# kdgan metric
#
################################################################

def compute_score(logits, labels, cutoff, normalize):
  predictions = np.argsort(-logits, axis=1)[:,:cutoff]
  batch_size, _ = labels.shape
  scores = []
  for batch in range(batch_size):
    label_bt = labels[batch,:]
    label_bt = np.nonzero(label_bt)[0]
    prediction_bt = predictions[batch,:]
    num_label = len(label_bt)
    present = 0
    for label in label_bt:
      if label in prediction_bt:
        present += 1
    score = present
    if score > 0:
      score *= (1.0 / normalize(cutoff, num_label))
    # print('score={0:.4f}'.format(score))
    scores.append(score)
  score = np.mean(scores)
  return score

def compute_prec(logits, labels, cutoff):
  def normalize(cutoff, num_label):
    return cutoff
  prec = compute_score(logits, labels, cutoff, normalize)
  # print('prec={0:.4f}'.format(prec))
  return prec

def compute_rec(logits, labels, cutoff):
  def normalize(cutoff, num_label):
    return num_label
  rec = compute_score(logits, labels, cutoff, normalize)
  # print('rec={0:.4f}'.format(rec))
  return rec

def compute_acc(predictions, labels):
  labels = np.argmax(labels, axis=1)
  acc = np.average(predictions == labels)
  return acc

def eval_mdlcompr(sess, vd_model, mnist):
  vd_image_np, vd_label_np = mnist.test.images, mnist.test.labels
  feed_dict = {vd_model.image_ph:vd_image_np}
  predictions, = sess.run([vd_model.predictions], feed_dict=feed_dict)
  acc_v = compute_acc(predictions, vd_label_np)
  return acc_v

def main():
  logits = np.log([
    [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
    [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
  ])
  labels = np.asarray([
    [0, 0, 0, 0], [0, 0, 0, 1],
    [0, 0, 1, 1], [1, 1, 1, 0],
  ], dtype=np.int32)
  cutoff = 2

  # prec = compute_hit(logits, labels, cutoff)
  # print('prec=%.4f' % (prec))

  # rec = compute_rec(logits, labels, cutoff)
  # print('rec=%.4f' % (rec))
  sorted_labels = [1, 1, 0, 0, 0]
  sorted_labels = [3, 2, 3, -1, 1, 2]
  nr_relevant = len([x for x in sorted_labels if x > 0])
      
  for scorer in [APScorer(0), APScorer(1), APScorer(2), APScorer(3), PrecisionScorer(1), PrecisionScorer(2), PrecisionScorer(10), NDCGScorer(10), RRScorer(0)]:
    print(scorer.name(), scorer.score(sorted_labels))

  
  sorted_labels = [3, 2, 3, 0, 1, 2]
  
  for k in range(1, 11):
    scorer1 = getScorer('NDCG@%d'%k)
    scorer2 = getScorer('NDCG2@%d'%k)
    print(k, scorer1.score(sorted_labels), scorer2.score(sorted_labels))

  predictions = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  labels = np.asarray([0, 1, 3, 3, 4, 5, 6, 8, 8, 9])
  acc = compute_acc(predictions, labels)
  print('acc=%.4f' % (acc))

if __name__ == '__main__':
    main()