import argparse
import numpy as np
import os
import pickle
import sys

from joblib import dump, load
from sklearn.linear_model import LogisticRegression


class Calibration:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Calibrate the scores given by boosting trees")
        parser.add_argument("task", help="select the task to perform, train or test")
        args = parser.parse_args(sys.argv[1:2])
        getattr(self, args.task)()

    def train(self):
        scores_file, model_file, result_file = get_args("train a calibration model", False)
        scores, labels = parse_scores(scores_file)
        lr = LogisticRegression()
        lr.fit(scores, labels)
        dump(lr, model_file)

    def test(self):
        scores_file, model_file, result_file = get_args(
                "get the calibrated probability from the boosting scores", True, True)
        scores, labels = parse_scores(scores_file)
        lr = load(model_file)
        proba = lr.predict_proba(scores)
        with open(result_file, "wb") as f:
            pickle.dump((proba, scores, labels), f)


def get_args(desc, model_req, result_req=False):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--scores", required=True, help="the file path to the scores file")
    parser.add_argument("--model", required=True, help="the file path to the model file")
    parser.add_argument("--result", required=result_req, help="the file path to write the calibrated scores")
    args = parser.parse_args(sys.argv[2:])
    scores_filepath = args.scores
    models_filepath = args.model
    result_filepath = args.result
    for req, filepath in [(True, scores_filepath), (model_req, models_filepath)]:
        if req and not os.path.exists(filepath):
            print("{} does not exist.".format(filepath))
            sys.exit(1)
    return scores_filepath, models_filepath, result_filepath


def parse_scores(filename):
    with open(filename, "rb") as f:
        _, labels, scores, _ = pickle.load(f)
    pos_scores = 1.0 - scores[labels == 0]
    neg_scores = 1.0 - scores[labels == 1]
    scores = np.concatenate([pos_scores, neg_scores]).reshape(-1, 1)
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    return (scores, labels)


if __name__ == "__main__":
    Calibration()
