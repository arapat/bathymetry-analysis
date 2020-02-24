import sys
import pickle
import numpy as np


def get_fnr_fixed_fpr(model_name, region_name, filename, fpr):
    with open(filename, "rb") as f:
        _, labels, scores, _ = pickle.load(f)
    pos_scores = scores[labels == 0]
    neg_scores = np.sort(scores[labels == 1])

    total_pos = pos_scores.shape[0]
    total_neg = neg_scores.shape[0]
    neg_rate = 1.0 * total_neg / (total_pos + total_neg)
    for rho in fpr:
        k = int(rho * len(neg_scores))  # neg should receive 1
        while k + 1 < len(neg_scores) and neg_scores[k + 1] - neg_scores[k] < 1e-8:
            k += 1
        fn = np.sum(pos_scores > neg_scores[k])
        fnr = 1.0 * fn / total_pos
        print("{} {} {} {} {} {} {}".format(
            model_name, region_name, total_pos, total_neg, neg_rate, rho, fnr))


if __name__ == "__main__":
    fpr = [0.001, 0.01, 0.1, 0.2, 0.3]
    get_fnr_fixed_fpr(sys.argv[1], sys.argv[2], sys.argv[3], fpr)

