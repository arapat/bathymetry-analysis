import json
import sys

from . import logger
from .train import run_training
from .test import run_testing
from .train_test import run_train_test


regions = ['AGSO', 'JAMSTEC', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']
usage_msg = "Usage: ./lgb.py <text|bin> <train|test|both> <config_path>"


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(usage_msg)
        sys.exit(1)
    if sys.argv[1].lower() not in ["text", "bin"] or \
            sys.argv[2].lower() not in ["train", "test", "both"]:
        print("Cannot understand the parameters.")
        print(usage_msg)
        sys.exit(1)
    is_read_text = (sys.argv[1].lower() == "text")
    with open(sys.argv[3]) as f:
        config = json.load(f)

    if sys.argv[2].lower() == "train":
        logger.set_file_handle("./training_log.log")
        run_training(config, regions, is_read_text)
    elif sys.argv[2].lower() == "test":
        logger.set_file_handle("./testing_log.log")
        run_testing(config, regions, is_read_text)
    else:  # "both"
        logger.set_file_handle("./train_test_log.log")
        run_train_test(config, regions, is_read_text)

