import json
import numpy as np
import pickle
import sys

from ..common import Logger
from ..load_data import get_region_data

TEST_PREFIX = "test"
TRAIN_PREFIX = "train"


regions = ['AGSO', 'JAMSTEC', 'NGA', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']


def get_all_data(base_dir, all_files, region, logger):
    is_read_text = False
    (features1, labels1, weights1) = get_region_data(base_dir, all_files, [region], is_read_text,
            TRAIN_PREFIX, logger)
    (features2, labels2, weights2) = get_region_data(base_dir, all_files, [region], is_read_text,
            TEST_PREFIX, logger)
    features = np.concatenate([features1, features2], axis=0)
    labels = np.concatenate([labels1, labels2], axis=0)
    weights = np.concatenate([weights1, weights2], axis=0)
    return (features, labels, weights)


def shuffle_and_save(features, labels, weights, region):
    assert(type(region) is str)
    order = np.arange(features.shape[0], dtype=int)
    np.random.shuffle(order)
    k = int(features.shape[0] * 0.8)
    with open("training-instances_{}.pkl".format(region), 'wb') as f:
        pickle.dump((features[order[:k]], labels[order[:k]], weights[order[:k]]), f, protocol=4)
    with open("testing-instances_{}.pkl".format(region), 'wb') as f:
        pickle.dump((features[order[k:]], labels[order[k:]], weights[order[k:]]), f, protocol=4)


def load_examples_from_pickle(pickle_files):
    features = []
    labels = []
    weights = []
    for filename in pickle_files:
        with open(filename, "rb") as f:
            _1, _2, _3 = pickle.load(f)
            features.append(_1)
            labels.append(_2)
            weights.append(_3)
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(weights, axis=0),
    )


def create_instance_based_splitting(config, regions, logger):
    with open(config["training_files"]) as f:
        all_training_files = f.readlines()
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()
    all_files = all_training_files + all_testing_files
    for region in regions:
        features, labels, weights = get_all_data(config["base_dir"], all_files, region, logger)
        shuffle_and_save(features, labels, weights, region)


if __name__ == "__main__":
    logger = Logger()
    logger.set_file_handle("splitting-by-instances.log")
    with open(sys.argv[1]) as f:
        config = json.load(f)
    create_instance_based_splitting(config, regions, logger)
