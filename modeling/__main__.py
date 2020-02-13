import json
import os
import ray
import sys

from .common import Logger
from .load_data import init_setup
from .train import run_training
from .train import run_training_all
from .test import get_all_data
from .test import run_testing


regions = ['AGSO', 'JAMSTEC', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']
param1 = ["text", "bin"]
param2 = ["train", "train-all", "test-self", "test-cross", "test-all"]
usage_msg = "Usage: ./lgb.py <{}> <{}> <config_path>".format("|".join(param1), "|".join(param2))


@ray.remote
def run_training_one_region(region):
    logfile = os.path.join(config["base_dir"], "training_log_{}.log".format(region))
    logger.set_file_handle(logfile)
    run_training(config, [region], is_read_text, logger)


def run_training_all_regions(regions):
    logfile = os.path.join(config["base_dir"], "training_log_all.log")
    logger.set_file_handle(logfile)
    run_training_all(config, regions, is_read_text, logger)


@ray.remote
def run_test(model_name, test_regions, task, data=None):
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "testing_log_{}.log".format(model_name))
    logger.set_file_handle(logfile)
    task = task.split('-')[1]
    run_testing(config, [model_name], test_regions, is_read_text, task, logger, all_data=data)


def get_data():
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "all-data-loading.log")
    logger.set_file_handle(logfile)
    logger.log("start loading all datasets")
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()
    data = get_all_data(config["base_dir"], all_testing_files, regions, is_read_text, logger)
    logger.log("finished loading all testing data")
    return data


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(usage_msg)
        sys.exit(1)
    if sys.argv[1].lower() not in param1 or sys.argv[2].lower() not in param2:
        print("Cannot understand the parameters.")
        print(usage_msg)
        sys.exit(1)
    with open(sys.argv[3]) as f:
        config = json.load(f)
    is_read_text = (sys.argv[1].lower() == "text")
    init_setup(config["base_dir"])
    task = sys.argv[2].lower()

    ray.init()
    result_ids = []
    if task == "train":
        for region in regions:
            result_ids.append(run_training_one_region.remote(region))
    elif task == "train-all":
        run_training_all_regions(regions)
    elif task == "test-self":
        for region in regions:
            result_ids.append(run_test.remote(region, [region], task))
    elif task == "test-cross":
        for region in regions:
            result_ids.append(run_test.remote(region, regions, task))
    elif task == "test-all":
        data = get_data()
        for region in regions:
            result_ids.append(run_test.remote(region, regions, task, data=data))
    else:
        assert(False)
    results = ray.get(result_ids)
