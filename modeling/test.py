import os
import pickle
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from .load_data import get_region_data
from .load_data import get_model_path
from .load_data import persist_predictions


TEST_PREFIX = "test"
LIMIT = None


def run_testing(config, regions, is_read_text, test_all, logger, fixed_model=None, all_data=None):
    base_dir = config["base_dir"]
    logger.log("start testing")
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()

    if test_all:
        model_name = "all"
        if fixed_model is not None:
            model_name = fixed_model
        run_testing_per_region(
            model_name, regions, base_dir, all_testing_files, is_read_text, logger, all_data)
    else:
        for region in regions:
            model_name = region
            if fixed_model is not None:
                model_name = fixed_model
            run_testing_per_region(
                model_name, region, base_dir, all_testing_files, is_read_text, logger)


def run_testing_per_region(
        model_region, test_region, base_dir, all_testing_files, is_read_text, logger, data=None):
    logger.log("start constructing datasets")
    if data is None:
        (features, labels, weights) = \
            get_region_data(base_dir, all_testing_files, test_region, is_read_text,
                    "{}_{}".format(TEST_PREFIX, model_region), LIMIT, logger)
    else:
        (features, labels, weights) = data
    logger.log("finished loading testing data")
    # Start training
    test_region_str = "all"
    if type(test_region) is str:
        test_region_str = test_region
    model_path = get_model_path(base_dir, model_region)
    scores = get_scores(model_region, test_region_str, features, labels, model_path, logger)
    persist_predictions(base_dir, model_region, test_region_str, features, labels, scores, weights)
    logger.log("finished testing")


def get_scores(region, test_region, features, labels, pkl_model_path, logger):
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

    logger.log("eval, {}, {}, {}, {}, {}, {}, {}".format(
        region, test_region, model.num_trees(), loss, auprc, auroc, acc))
    return scores


def get_all_data(base_dir, all_testing_files, test_region, is_read_text, logger):
    return get_region_data(base_dir, all_testing_files, test_region, is_read_text,
            "{}_{}".format(TEST_PREFIX, "all"), LIMIT, logger)
