import json
import ray
import sys

from .common import Logger
from .load_data import init_setup
from .train import run_training
from .test import run_testing
from .train_test import run_train_test


all_regions = ['AGSO', 'JAMSTEC', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']
train_all = False
usage_msg = "Usage: ./lgb.py <text|bin> <train|test|both> <config_path>"


@ray.remote
def run_prog(regions, task):
    logger = Logger()
    if task == "train":
        logger.set_file_handle("./training_log_{}.log".format(regions[0]))
        run_training(config, regions, is_read_text, train_all, logger)
    elif task == "test":
        logger.set_file_handle("./testing_log_{}.log".format(regions[0]))
        run_testing(config, regions, is_read_text, train_all, logger)
    else:  # "both"
        logger.set_file_handle("./train_test_log_{}.log".format(regions[0]))
        run_train_test(config, regions, is_read_text, train_all, logger)


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
    init_setup(config["base_dir"])

    # with open("regions.txt") as f:
    #     regions = f.readline().strip().split()
    ray.init()
    task = sys.argv[2].lower()
    result_ids = []
    for region in all_regions:
        result_ids.append(run_prog.remote([region], task))
    results = ray.get(result_ids)

