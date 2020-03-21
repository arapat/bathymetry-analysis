import os
import pickle
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from .booster import get_scores
from .load_data import get_region_data
from .load_data import get_model_path
from .load_data import persist_predictions
from .tools.split_by_instances import load_examples_from_pickle


TEST_PREFIX = "test"
LIMIT = None


def run_testing(config, models, regions, is_read_text, test_mode, logger, all_data=None):
    base_dir = config["base_dir"]
    logger.log("start testing")
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()

    assert(test_mode in ["self", "cross", "all"])
    if test_mode == "self":
        assert(len(models) == len(regions))
        for model_name, region in zip(models, regions):
            run_testing_per_region(
                model_name, [region], region, base_dir, all_testing_files, is_read_text, logger,
                all_data)
    elif test_mode == "cross":
        for model_name in models:
            for region in regions:
                run_testing_per_region(
                    model_name, [region], region, base_dir, all_testing_files, is_read_text, logger,
                    all_data)
    else:  # test_mode == "all"
        for model_name in models:
            run_testing_per_region(
                model_name, regions, "all", base_dir, all_testing_files, is_read_text, logger,
                all_data)


def run_testing_per_region(
        model_region, test_regions, test_region_str, base_dir, all_testing_files, is_read_text,
        logger, data=None):
    logger.log("start constructing datasets")
    if data is None:
        (features, labels, weights) = \
            get_region_data(base_dir, all_testing_files, test_regions, is_read_text,
                    TEST_PREFIX, logger)
    else:
        (features, labels, weights) = data
    logger.log("finished loading testing data")
    # Start training
    model_path = get_model_path(base_dir, model_region)
    scores = get_scores(model_region, test_region_str, features, labels, model_path, logger)
    persist_predictions(base_dir, model_region, test_region_str, features, labels, scores, weights)
    logger.log("finished testing")


def get_all_data(base_dir, all_files, test_regions, is_read_text, logger):
    return get_region_data(base_dir, all_files, test_regions, is_read_text, TEST_PREFIX, logger)


# Specify a data file
def run_testing_specific_file(model_name, test_filenames, test_region_name, config, logger):
    logger.log("start loading datasets")
    features, labels, weights = load_examples_from_pickle(test_filenames)
    logger.log("finished loading testing data")
    model_path = get_model_path(config["base_dir"], model_name)
    scores = get_scores(model_name, test_region_name, features, labels, model_path, logger)
    persist_predictions(
        config["base_dir"], model_name, test_region_name, features, labels, scores, weights)
