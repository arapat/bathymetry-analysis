import sys
from time import time

from .common import Logger
from .train import run_training
from .test import run_testing
from .common import TRAINING_FILES_DESC
from .common import VALIDATION_FILES_DESC
from .common import TESTING_FILES_DESC


regions = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']
config = {
    "base_dir": "./",
    "training_files": TRAINING_FILES_DESC,
    "validation_files": VALIDATION_FILES_DESC,
    "testing_files": TESTING_FILES_DESC,
    "num_leaves": 31,
    "rounds": 1000,
    "early_stopping_rounds": 1000,
    "max_bin": 255,
}
usage_msg = "Usage: ./lgb.py <train|test|both> <text|bin>"

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(usage_msg)
        sys.exit(1)
    if sys.argv[1].lower() not in ["train", "test", "both"] or \
            sys.argv[2].lower() not in ["text", "bin"]:
        print("Cannot understand the parameters.")
        print(usage_msg)
        sys.exit(1)
    is_read_text = sys.argv[2].lower()

    if sys.argv[1].lower() in ["both", "train"]:
        logger = Logger("./training_log.log")
        run_training(config, regions, is_read_text)
    if sys.argv[1].lower() in ["both", "test"]:
        logger = Logger("./testing_log.log")
        run_testing(config, regions, is_read_text)
