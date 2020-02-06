import multiprocessing
import lightgbm as lgb

from .common import print_ts
from .load_data import get_region_data
from .load_data import persist_model


TRAIN_PREFIX = "train"
VALID_PREFIX = "valid"
LIMIT = None


def run_training_per_region(config, region, all_training_files, all_valid_files, is_read_text, logger):
    logger.log("Now training {}".format(region))

    logger.log("start constructing datasets")
    region_str = "all"
    if type(region) is not list:
        region_str = region
    (t_features, t_labels, t_weights) = get_region_data(
        config["base_dir"], all_training_files, region, is_read_text,
        "{}_{}".format(TRAIN_PREFIX, region_str), LIMIT, logger)
    train_dataset = lgb.Dataset(
        t_features, label=t_labels, weight=t_weights, params={'max_bin': config["max_bin"]})
    (v_features, v_labels, v_weights) = get_region_data(
        config["base_dir"], all_valid_files, region, is_read_text,
        "{}_{}".format(VALID_PREFIX, region_str), LIMIT, logger)
    valid_dataset = lgb.Dataset(
        v_features, label=v_labels, weight=v_weights, params={'max_bin': config["max_bin"]})

    logger.log("start training...")
    # Strange bug exists that prevents saving all iterations if `early_stopping_rounds` is enabled
    config["early_stopping_rounds"] = None
    gbm = lgb.train(
        config,
        train_dataset,
        num_boost_round=config["rounds"],
        valid_sets=[train_dataset, valid_dataset],
        callbacks=[print_ts(logger)],
        # early_stopping_rounds=config["early_stopping_rounds"],
        # fobj=expobj, feval=exp_eval,
    )
    logger.log("training completed.")
    persist_model(config["base_dir"], region_str, gbm)
    logger.log("Model for {} is persisted".format(region_str))


def run_training(config, regions, is_read_text, train_all, logger):
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    if train_all:
        run_training_per_region(config, regions, all_training_files, all_valid_files, is_read_text,
                logger)
    else:
        for region in regions:
            run_training_per_region(config, region, all_training_files, all_valid_files,
                    is_read_text, logger)
