import numpy as np

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

def compute_hit(logits, labels, cutoff):
    def normalize(cutoff, num_label):
        return min(cutoff, num_label)
    hit = compute_score(logits, labels, cutoff, normalize)
    # print('hit={0:.4f}'.format(hit))
    return hit

def compute_rec(logits, labels, cutoff):
    def normalize(cutoff, num_label):
        return num_label
    rec = compute_score(logits, labels, cutoff, normalize)
    # print('rec={0:.4f}'.format(rec))
    return rec

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
    hit = compute_hit(logits, labels, cutoff)
    rec = compute_rec(logits, labels, cutoff)
    print('hit={0:.4f} rec={1:.4f}'.format(hit, rec))

if __name__ == '__main__':
    main()