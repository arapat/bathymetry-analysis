import sys
import pickle
import numpy as np


def get_recall_fixed_fdr(model_name, region_name, filename, fdr):
    with open(filename, "rb") as f:
        _, labels, scores, _ = pickle.load(f)
    pos_scores = 1.0 - scores[labels == 0]
    neg_scores = 1.0 - scores[labels == 1]
    total_pos = pos_scores.shape[0]
    total_neg = neg_scores.shape[0]
    neg_rate = 1.0 * total_neg / (total_pos + total_neg)

    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    labels = labels[order]
    false_discovery = np.cumsum(1.0 - labels) / np.arange(1.0, labels.shape[0] + 1)
    for i in range(false_discovery.shape[0] - 1, 0, -1):
        if scores[i - 1] - scores[i] < 1e-8:
            false_discovery[i - 1] = false_discovery[i]

    false_discovery = false_discovery[::-1]
    scores = scores[::-1]
    for rho in fdr:
        index = np.argmax(false_discovery <= rho)
        if false_discovery[index] > rho:
            recall = 0.0
        else:
            recall_num = np.sum(pos_scores >= scores[index])
            recall = 1.0 * recall_num / total_pos
        print("{} {} {} {} {} {} {} {}".format(
            model_name, region_name, total_pos, total_neg, neg_rate, rho, index, recall))


if __name__ == "__main__":
    fdr = [0.0001, 0.001, 0.01, 0.02, 0.05]
    get_recall_fixed_fdr(sys.argv[1], sys.argv[2], sys.argv[3], fdr)

