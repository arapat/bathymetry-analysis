import os
import pickle
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from .load_data import get_region_data
from .load_data import get_model_path
from .load_data import get_prediction_path
from .load_data import persist_predictions


TEST_PREFIX = "test"
LIMIT = None


def run_testing(config, regions, is_read_text, test_all, logger):
    base_dir = config["base_dir"]
    logger.log("start testing")
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()

    if test_all:
        run_testing_per_region(regions, base_dir, all_testing_files, is_read_text, logger)
    else:
        for region in regions:
            run_testing_per_region(region, base_dir, all_testing_files, is_read_text, logger)


def run_testing_per_region(region, base_dir, all_testing_files, is_read_text, logger):
    logger.log("start constructing datasets")
    region_str = "all"
    if type(region) is not list:
        region_str = region
    (features, labels, weights) = \
        get_region_data(base_dir, all_testing_files, region, is_read_text,
                "{}_{}".format(TEST_PREFIX, region_str), LIMIT, logger)
    logger.log("finished loading testing data")
    # Start training
    model_path = get_model_path(base_dir, region_str)
    scores = get_scores(region_str, features, labels, model_path)
    persist_predictions(base_dir, region_str, features, labels, scores, weights)
    logger.log("finished testing")


def get_scores(region, features, labels, pkl_model_path):
    # load model with pickle to predict
    with open(pkl_model_path, 'rb') as fin:
        model = pickle.load(fin)

    # Prediction
    preds = model.predict(features)
    scores = np.clip(preds, 1e-15, 1.0 - 1e-15)
    logger.log('finished prediction')

    # compute auprc
    loss = np.mean(labels * -np.log(scores) + (1 - labels) * -np.log(1.0 - scores))
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    # accuracy
    acc = np.sum(labels == (scores > 0.5)) / labels.shape[0]

    logger.log("eval, {}, {}, {}, {}, {}, {}".format(
        region, model.num_trees(), loss, auprc, auroc, acc))
    return scores
