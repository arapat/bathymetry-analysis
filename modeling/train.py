import multiprocessing
import lightgbm as lgb

from .load_data import get_region_data
TRAIN_PREFIX = "train"
VALID_PREFIX = "valid"
LIMIT = None


def run_training_per_region(
        config, regions, region_str, all_training_files, all_valid_files, is_read_text, logger):
    logger.log("Now training {}".format(region_str))

    logger.log("start constructing datasets")
    (t_features, t_labels, t_weights) = get_region_data(
        config["base_dir"], all_training_files, regions, is_read_text,
        "{}_{}".format(TRAIN_PREFIX, region_str), LIMIT, logger)
    train_dataset = lgb.Dataset(
        t_features, label=t_labels, weight=t_weights, params={'max_bin': config["max_bin"]})
    (v_features, v_labels, v_weights) = get_region_data(
        config["base_dir"], all_valid_files, regions, is_read_text,
        "{}_{}".format(VALID_PREFIX, region_str), LIMIT, logger)
    valid_dataset = lgb.Dataset(
        v_features, label=v_labels, weight=v_weights, params={'max_bin': config["max_bin"]})

    booster.train(config, train_dataset, valid_dataset, region_str, logger)


def run_training(config, regions, is_read_text, logger):
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    for region in regions:
        run_training_per_region(
            config, [region], region, all_training_files, all_valid_files, is_read_text, logger)


def run_training_all(config, regions, is_read_text, logger):
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    run_training_per_region(
        config, regions, "all", all_training_files, all_valid_files, is_read_text, logger)