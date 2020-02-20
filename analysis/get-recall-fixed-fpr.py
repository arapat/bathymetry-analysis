import sys
import numpy as np


def get_recall_fixed_fpr(model_name, region_name, filename, fpr):
    with open(filename, "rb") as f:
        _, labels, scores, _ = pickle.load(f)
    neg_scores = np.sort(scores[labels == 0])[::-1]
    pos_scores = scores[labels == 1]
    total_pos = pos_scores.shape[0]
    total_neg = neg_scores.shape[0]
    pos_rate = 1.0 * total_pos / (total_pos + total_neg)
    for rho in fpr:
        k = rho * len(neg_scores)
        while k + 1 < len(neg_scores) and neg_scores[k] - neg_scores[k + 1] < 1e-8:
            k += 1
        recall = np.sum(pos_scores >= neg_scores[k])
        rate = 1.0 * recall / total_pos
        print("{} {} {} {} {} {} {}".format(
            model_name, region_name, total_pos, total_neg, pos_rate, rho, rate))


if __name__ == "__main__":
    fpr = [0.0001, 0.001, 0.01, 0.02]
    get_recall_fixed_fpr(sys.argv[1], sys.argv[2], sys.argv[3], fpr)
