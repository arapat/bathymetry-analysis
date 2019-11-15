import io
import os
import pickle
import numpy as np
from time import time
from . import logger


DEBUG = True

NUM_COLS = 35
TYPE_INDEX = 17
REMOVED_FEATURES = [3, 4, 5, 7]

MAX_NUM_EXAMPLES_PER_PICKLE = 1000000
if DEBUG:
    MAX_NUM_EXAMPLES_PER_PICKLE = 300000
MAX_WEIGHT = 1.0 / 100000.0
BINARY_DIR = "runtime_data"
MODEL_DIR = "runtime_models"
SCORES_DIR = "runtime_scores"
INVENTORY = os.path.join(BINARY_DIR, "inventory.txt")

data_type = {
    "M": 1,  # - multibeam
    "G": 2,  # - grid
    "S": 3,  # - single beam
    "P": 4,  # - point measurement
}


def read_data_from_text(filename, get_label=lambda cols: cols[4] == '9999'):
    features = []
    labels = []
    filename = filename.strip()
    incorrect_cols = 0
    with io.open(filename, 'r', newline='\n') as fread:
        for line in fread:
            cols = line.strip().split()
            if len(cols) != NUM_COLS:
                incorrect_cols += 1
                continue
            cols[TYPE_INDEX] = data_type[cols[TYPE_INDEX]]
            labels.append(get_label(cols))
            features.append(np.array(
                [float(cols[i]) for i in range(len(cols)) if i not in REMOVED_FEATURES]
            ))
    assert(len(features) == len(labels))
    weights = np.ones_like(labels) * max(MAX_WEIGHT, 1.0 / max(1.0, len(labels)))
    return (features, labels, weights.tolist(), incorrect_cols)


def read_data_from_binary(filename):
    with open(filename, 'rb') as f:
        features, labels, weights = pickle.load(f)
    return (features, labels, weights, 0)


def write_data_to_binary(st, features, labels, weights, filename):
    with open(filename, 'wb') as f:
        pickle.dump((features[st:], labels[st:], weights[st:]), f, protocol=4)
    with open(INVENTORY, 'a') as f:
        f.write(filename.strip() + '\n')


def get_datasets(filepaths, is_read_text, prefix="", limit=None): 
    data_features = []
    data_labels = []
    data_weights = []
    last_written_length = 0
    inventory = load_inventory(is_read_text)
    start_time = time()
    for filename in filepaths:
        filename = filename.strip()
        bin_filename = get_binary_filename(prefix, filename)
        if not is_read_text:
            filename = bin_filename
        try:
            if is_read_text:
                features, labels, weights, incorrect_cols = read_data_from_text(filename)
            else:
                features, labels, weights, incorrect_cols = read_data_from_binary(filename)
            logger.log("loaded, {}, incorrect cols, {}, size, {}".format(
                filename, incorrect_cols, len(features)))
        except Exception as err:
            # Print error message only if we are supposed to read this file
            if is_read_text or filename in inventory:
                logger.log("Failed to load {}, is_read_text, {}, Error, {}".format(
                    filename, is_read_text, err))
            continue

        data_features  += features
        data_labels    += labels
        data_weights   += weights

        curr_num_examples = len(data_features)
        if is_read_text and curr_num_examples - last_written_length >= MAX_NUM_EXAMPLES_PER_PICKLE:
            logger.log("To write {} examples".format(curr_num_examples - last_written_length))
            write_data_to_binary(
                last_written_length, data_features, data_labels, data_weights, bin_filename)
            last_written_length = curr_num_examples
        if limit is not None and curr_num_examples > limit or DEBUG and time() - start_time > 10:
            break

    # Handle last batch
    curr_num_examples = len(data_features)
    if is_read_text and curr_num_examples > last_written_length:
        logger.log("To write {} examples".format(curr_num_examples - last_written_length))
        write_data_to_binary(
            last_written_length, data_features, data_labels, data_weights, bin_filename)
    # Format labels and weights
    data_features = np.array(data_features)
    data_labels   = (np.array(data_labels) > 0).astype(np.int8)
    data_weights  = np.array(data_weights)
    logger.log("Dataset is loaded, size {}".format(data_features.shape))
    return (data_features, data_labels, data_weights)


def get_binary_filename(prefix, filename):
    if prefix and not prefix.endswith('_'):
        prefix = prefix + '_'
    if not os.path.exists(BINARY_DIR):
        os.mkdir(BINARY_DIR)
    basename = os.path.basename(filename)
    dirname = os.path.basename(os.path.dirname(filename))
    filename = prefix + dirname + '_' + basename + ".pkl"
    return os.path.join(BINARY_DIR, filename)


def get_region_data(files, region, is_read_text, prefix, limit):
    region_files = [filepath for filepath in files if "/{}/".format(region) in filepath]
    return get_datasets(region_files, is_read_text, prefix=prefix, limit=limit)


def get_model_path(base_dir, region):
    dir_path = os.path.join(base_dir, MODEL_DIR)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return os.path.join(dir_path, '{}_model.pkl'.format(region))


def get_prediction_path(base_dir, region):
    dir_path = os.path.join(base_dir, SCORES_DIR)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return os.path.join(dir_path, '{}_scores.pkl'.format(region))


def persist_predictions(base_dir, region, label, scores, weights):
    with open(get_prediction_path(base_dir, region), 'wb') as fout:
        pickle.dump((label, scores, weights), fout)


def persist_model(base_dir, region, gbm):
    pkl_model_path = get_model_path(base_dir, region)
    txt_model_path = pkl_model_path.rsplit('.', 1)[0] + ".txt"
    with open(pkl_model_path, 'wb') as fout:
        pickle.dump(gbm, fout)
    gbm.save_model(txt_model_path)


def load_inventory(is_read_text):
    if is_read_text:
        if not os.path.exists(BINARY_DIR):
            os.mkdir(BINARY_DIR)
        f = open(INVENTORY, 'w')
        f.close()
        return []
    with open(INVENTORY) as f:
        return [line.strip() for line in f.readlines()]

