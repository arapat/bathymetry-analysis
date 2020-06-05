import io
import os
import pickle
import numpy as np
from time import time


DEBUG = False

NUM_COLS = 36
TYPE_INDEX = 35
REMOVED_FEATURES = [3, 4, 5, 7]
REMOVED_FEATURES_FROM_BIN = []

MAX_NUM_EXAMPLES_PER_PICKLE = 1000000
if DEBUG:
    MAX_NUM_EXAMPLES_PER_PICKLE = 300000
MAX_WEIGHT = 1.0 / 100000.0
BINARY_DIR = "runtime_data"
MODEL_DIR = "runtime_models"
SCORES_DIR = "runtime_scores"
INVENTORY = os.path.join(BINARY_DIR, "inventory_{}.txt")

data_type = {
    "M": 1,  # - multibeam
    "G": 2,  # - grid
    "S": 3,  # - single beam
    "P": 4,  # - point measurement
    "X": np.nan,
    'nan': np.nan,
}

inst_weights = {
    'AGSO': 13760454.0,
    'JAMSTEC': 64869049.0,
    'JAMSTEC2': 4467504.0,
    'NGA': 3521481.0,
    'NGA2': 7060849.0,
    'NGDC': 86229443.0,
    'NOAA_geodas': 28041621.0,
    'SIO': 30369189.0,
    'US_multi': 35984658.0,
}


def init_setup(base_dir):
    for dirname in [BINARY_DIR, MODEL_DIR, SCORES_DIR]:
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


# cols[4] == 9999, the instance is corrupted, set label to 0
# otherwise, the instance is good, set the label to 1
def read_data_from_text(filename, get_label=lambda cols: cols[4] != '9999'):
    features = []
    labels = []
    filename = filename.strip()
    incorrect_cols = 0
    with io.open(filename, 'r', newline='\n') as fread:
        for line in fread:
            cols = line.strip().split()
            if len(cols) not in [NUM_COLS, NUM_COLS - 2]:
                incorrect_cols += 1
                continue
            if len(cols) == NUM_COLS - 2:
                cols = ["X"] * 2
            cols[TYPE_INDEX] = data_type[cols[TYPE_INDEX]]
            labels.append(get_label(cols))
            features.append(np.array(
                [float(cols[i]) for i in range(len(cols)) if i not in REMOVED_FEATURES]
            ))
    assert(len(features) == len(labels))
    # weights = np.ones_like(labels) * max(MAX_WEIGHT, 1.0 / max(1.0, len(labels)))
    weights = np.ones_like(labels)
    return (features, labels, weights.tolist(), incorrect_cols)


def read_data_from_binary(filename):
    with open(filename, 'rb') as f:
        features, labels, weights = pickle.load(f)
    return (features, labels, weights, 0)


def write_data_to_binary(base_dir, st, features, labels, weights, filename, prefix):
    with open(filename, 'wb') as f:
        pickle.dump((features[st:], labels[st:], weights[st:]), f, protocol=4)
    inv_filename = os.path.join(base_dir, INVENTORY.format(prefix))
    with open(inv_filename, 'a') as f:
        f.write(filename.strip() + '\n')


def get_datasets(region_str, base_dir, filepaths, is_read_text, prefix, logger):
    data_features = []
    data_labels = []
    data_weights = []
    source_filename = []
    last_written_length = 0
    for filename in filepaths:
        filename = filename.strip()
        bin_filename = get_binary_filename(base_dir, prefix, filename)
        if not is_read_text:
            filename = bin_filename
        try:
            if is_read_text:
                features, labels, weights, incorrect_cols = read_data_from_text(filename)
            else:
                features, labels, weights, incorrect_cols = read_data_from_binary(filename)
            logger.log("loaded, {}, incorrect cols, {}, corrupt, {}, size, {}, dim, {}".format(
                filename, incorrect_cols, np.sum(labels), len(features), features[0].shape[0]))
        except Exception as err:
            # Print error message only if we are supposed to read this file
            logger.log("Failed to load {}, is_read_text, {}, Error, {}".format(
                filename, is_read_text, err))
            continue

        region_name = None
        for t in inst_weights:
            if t in filename:
                region_name = t
                break
        weights = np.ones_like(weights) / inst_weights[region_name]
        data_features  += features
        data_labels    += labels
        data_weights   += weights.tolist()
        source_filename += [filename] * len(features)

        curr_num_examples = len(data_features)
        if is_read_text:  # and curr_num_examples - last_written_length >= MAX_NUM_EXAMPLES_PER_PICKLE:
            logger.log("To write {} examples".format(curr_num_examples - last_written_length))
            write_data_to_binary(
                base_dir, last_written_length, data_features, data_labels, data_weights,
                bin_filename, prefix)
            last_written_length = curr_num_examples

    # Format labels and weights
    data_features = np.array(data_features)
    data_labels   = (np.array(data_labels) > 0).astype(np.int8)
    data_weights  = np.array(data_weights)
    with open("sources-{}.txt".format(region_str), "w") as f:
        f.write("\n".join(source_filename))
    # Remove unwanted features when reading from the binary form
    if not is_read_text:
        mask = np.ones(shape=data_features.shape[1]).astype(bool)
        for i in REMOVED_FEATURES_FROM_BIN:
            mask[i] = False
        data_features = data_features[:, mask]
    logger.log("Dataset is loaded, size {}".format(data_features.shape))
    return (data_features, data_labels, data_weights)


def get_binary_filename(base_dir, prefix, filename):
    if filename.endswith(".pkl"):
        return filename
    while prefix and prefix.endswith('_'):
        prefix = prefix[:-1]
    basename = os.path.basename(filename)
    dirname = os.path.basename(os.path.dirname(filename))
    filename = prefix + '_' + dirname + '_' + basename + ".pkl"
    return os.path.join(base_dir, os.path.join(BINARY_DIR, filename))


def get_region_data(base_dir, files, regions, is_read_text, prefix, logger):
    def get_files(region):
        return [filepath for filepath in files
                if "/{}/".format(region) in filepath]
    assert(type(regions) is list)
    region_files = []
    for t in regions:
        region_files += get_files(t)
    return get_datasets(regions[0], base_dir, region_files, is_read_text, prefix, logger)


def get_model_path(base_dir, region):
    dir_path = os.path.join(base_dir, MODEL_DIR)
    return os.path.join(dir_path, '{}_model.pkl'.format(region))


def get_prediction_path(base_dir, model_region, test_region):
    dir_path = os.path.join(base_dir, SCORES_DIR)
    return os.path.join(dir_path, 'model_{}_test_{}_scores.pkl'.format(model_region, test_region))


def persist_predictions(base_dir, model_region, test_region, features, label, scores, weights):
    with open(get_prediction_path(base_dir, model_region, test_region), 'wb') as fout:
        pickle.dump((features[:, :4], label, scores, weights), fout)


def persist_model(base_dir, region, gbm):
    pkl_model_path = get_model_path(base_dir, region)
    txt_model_path = pkl_model_path.rsplit('.', 1)[0] + ".txt"
    with open(pkl_model_path, 'wb') as fout:
        pickle.dump(gbm, fout)
    gbm.save_model(txt_model_path)


def _load_inventory(base_dir, is_read_text, prefix):
    if is_read_text:
        filename = os.path.join(base_dir, INVENTORY.format(prefix))
        f = open(filename, 'w')
        f.close()
        return []
    filename = os.path.join(base_dir, INVENTORY.format(prefix))
    with open(filename) as f:
        return [line.strip() for line in f.readlines()]
