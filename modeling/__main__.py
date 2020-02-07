import json
import os
import ray
import sys

from .common import Logger
from .load_data import init_setup
from .train import run_training
from .test import run_testing
from .train_test import run_train_test


all_regions = ['AGSO', 'JAMSTEC', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']
RUN_ALL = True
usage_msg = "Usage: ./lgb.py <text|bin> <train|test|both> <config_path>"


@ray.remote
def run_prog(regions, task, test_model=None):
    logger = Logger()
    if task == "train":
        logfile = os.path.join(config["base_dir"], "training_log_{}.log".format(regions[0]))
        logger.set_file_handle(logfile)
        run_training(config, regions, is_read_text, RUN_ALL, logger)
    elif task == "test":
        logfile = os.path.join(config["base_dir"], "testing_log_{}.log".format(regions[0]))
        logger.set_file_handle(logfile)
        run_testing(config, regions, is_read_text, RUN_ALL, logger)
    elif task == "cross-test":
        assert(test_model is not None)
        logfile = os.path.join(config["base_dir"], "cross_testing_log_{}.log".format(test_model))
        logger.set_file_handle(logfile)
        run_testing(config, regions, is_read_text, False, logger, fixed_model=test_model)
        run_testing(config, regions, is_read_text, True, logger, fixed_model=test_model)
    else:  # "both"
        logfile = os.path.join(config["base_dir"], "train_test_log_{}.log".format(regions[0]))
        logger.set_file_handle(logfile)
        run_train_test(config, regions, is_read_text, RUN_ALL, logger)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(usage_msg)
        sys.exit(1)
    if sys.argv[1].lower() not in ["text", "bin"] or \
            sys.argv[2].lower() not in ["train", "test", "cross-test", "both"]:
        print("Cannot understand the parameters.")
        print(usage_msg)
        sys.exit(1)
    is_read_text = (sys.argv[1].lower() == "text")
    with open(sys.argv[3]) as f:
        config = json.load(f)
    init_setup(config["base_dir"])

    task = sys.argv[2].lower()
    ray.init()
    if RUN_ALL and task != "cross-test":
        result_id = run_prog.remote(all_regions, task)
        ray.get([result_id])
    else:
        result_ids = []
        if task == "cross-test":
            for region in all_regions:
                result_ids.append(run_prog.remote(all_regions, task, region))
            result_ids.append(run_prog.remote(all_regions, task, "all"))
        else:
            for region in all_regions:
                result_ids.append(run_prog.remote([region], task))
        results = ray.get(result_ids)

