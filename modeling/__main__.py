import sys

from . import config
from . import logger
from .train import run_training
from .test import run_testing


regions = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']
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
    is_read_text = (sys.argv[2].lower() == "text")

    if sys.argv[1].lower() in ["both", "train"]:
        logger.set_file_handle("./training_log.log")
        run_training(config, regions, is_read_text)
    if sys.argv[1].lower() in ["both", "test"]:
        logger.set_file_handle("./testing_log.log")
        run_testing(config, regions, is_read_text)

