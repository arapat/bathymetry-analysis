import multiprocessing
import lightgbm as lgb

from . import logger
from .common import print_ts
from .load_data import get_region_data
from .load_data import persist_model


TRAIN_PREFIX = "train"
VALID_PREFIX = "valid"
LIMIT = None


def run_training_per_region(config, region, all_training_files, all_valid_files, is_read_text):
    logger.log("Now training {}".format(region))

    logger.log("start constructing datasets")
    (t_features, t_labels, t_weights) = get_region_data(
        all_training_files, region, is_read_text, "{}_{}".format(TRAIN_PREFIX, region), LIMIT)
    train_dataset = lgb.Dataset(
        t_features, label=t_labels, weight=t_weights, params={'max_bin': config["max_bin"]})
    (v_features, v_labels, v_weights) = get_region_data(
        all_valid_files, region, is_read_text, "{}_{}".format(VALID_PREFIX, region), LIMIT)
    valid_dataset = lgb.Dataset(
        v_features, label=v_labels, weight=v_weights, params={'max_bin': config["max_bin"]})

    logger.log("start training...")
    gbm = lgb.train(
        config,
        train_dataset,
        num_boost_round=config["rounds"],
        early_stopping_rounds=config["early_stopping_rounds"],
        valid_sets=[train_dataset, valid_dataset],
        callbacks=[print_ts()],
        # fobj=expobj, feval=exp_eval,
    )
    logger.log("training completed.")
    persist_model(config["base_dir"], region, gbm)
    logger.log("Model for {} is persisted".format(region))


def run_training(config, regions, is_read_text):
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    for region in regions:
        run_training_per_region(config, region, all_training_files, all_valid_files, is_read_text)
