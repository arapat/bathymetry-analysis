from . import logger
from .train import run_training_per_region
from .test import run_testing_per_region


def run_train_test(config, regions, is_read_text):
    logger.log("start training and testing")
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["validation_files"]) as f:
        all_valid_files = f.readlines()
    base_dir = config["base_dir"]

    for region in regions:
        run_training_per_region(config, region, all_training_files, all_valid_files, is_read_text)
        run_testing_per_region(region, base_dir, all_testing_files, is_read_text)

