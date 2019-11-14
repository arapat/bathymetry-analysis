import os
import pickle
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from modeling import logger
from .load_data import get_region_data
from .load_data import get_model_path
from .load_data import get_prediction_path
from .load_data import persist_predictions


TEST_PREFIX = "test"
LIMIT = 3000


def run_testing(config, regions, is_read_text):
    base_dir = config["base_dir"]
    logger.log("start testing")
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()

    run_testing_per_region(regions, base_dir, all_testing_files, is_read_text)


def run_testing_per_region(regions, base_dir, all_testing_files, is_read_text):
    for region in regions:
        logger.log("start constructing datasets")
        (features, labels, weights) = \
            get_region_data(all_testing_files, region, is_read_text, TEST_PREFIX, LIMIT)
        logger.log("finished loading testing data")
        # Start training
        model_path = get_model_path(base_dir, region)
        scores = get_scores(features, labels, model_path)
        persist_predictions(base_dir, region, labels, scores, weights)
        logger.log("finished testing")


def get_scores(features_test, labels_test, pkl_model_path):
    # format raw testing input
    features = np.concatenate(features_test, axis=0)
    true = labels_test * 1

    # load model with pickle to predict
    with open(pkl_model_path, 'rb') as fin:
        model = pickle.load(fin)

    # Prediction
    preds = model.predict(features)
    scores = np.clip(preds, 1e-15, 1.0 - 1e-15)
    logger.log('finished prediction')

    # compute auprc
    loss = np.mean(true * -np.log(scores) + (1 - true) * -np.log(1.0 - scores))
    precision, recall, _ = precision_recall_curve(true, scores, pos_label=1)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    # accuracy
    acc = np.sum(true == (scores > 0.5)) / true.shape[0]

    logger.log("eval, {}, {}, {}, {}, {}".format(model.num_trees(), loss, auprc, auroc, acc))
    return scores
