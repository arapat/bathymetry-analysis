#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sys
import pickle
import lightgbm as lgb
import multiprocessing
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from time import time


# In[12]:
DATA_TYPE = {
    "M": 1,  # - multibeam
    "G": 2,  # - grid
    "S": 3,  # - single beam
    "P": 4,  # - point measurement
}


def logger(s, show_time=False):
    if show_time:
        msg = "Current time: %.2f" % time()
        print(msg)
        flog.write(msg + '\n')
    msg = "[%.5f] %s" % (time() - t0, s)
    print(msg)
    flog.write(msg + '\n')
    sys.stdout.flush()
    flog.flush()


def print_ts(period=1):
    """Create a callback that prints the tree is generated.

    Parameters
    ----------
    period : int, optional (default=1)
        The period to print the evaluation results.

    Returns
    -------
    callback : function
        The callback that prints the evaluation results every ``period`` iteration(s).
    """
    def callback(env):
        """internal function"""
        if period > 0 and (env.iteration + 1) % period == 0:
            logger('Tree %d' % (env.iteration + 1))
    callback.order = 10
    return callback


# In[ ]:
def train_lgb(train_data, valid_data, num_leaves, rounds, early_stopping_rounds, max_bin):
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',  # 'goss',
        'num_leaves': num_leaves,
        'learning_rate': 0.1,
        'tree_learner': 'serial',
        'task': 'train',
        'num_thread': multiprocessing.cpu_count(),
        'min_data_in_leaf': 5000,  # This is min bound for stopping rule in Sparrow
        'two_round': True,
        'is_unbalance': True,
        'max_bin': max_bin,
    }

    logger('Start training...')
    gbm = lgb.train(
        params, train_data, num_boost_round=rounds,
        early_stopping_rounds=early_stopping_rounds,
        # fobj=expobj, feval=exp_eval,
        valid_sets=[train_data, valid_data], callbacks=[print_ts()],
    )
    logger('Training completed.')
    return gbm


def persist_model(base_dir, gbm):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    txt_model_path = os.path.join(base_dir, 'model.txt')
    pkl_model_path = os.path.join(base_dir, 'model.pkl')
    gbm.save_model(txt_model_path)
    with open(pkl_model_path, 'wb') as fout:
        pickle.dump(gbm, fout)
    return (txt_model_path, pkl_model_path)


# In[ ]:


def run_test(features_test, labels_test, pkl_model_path):
    # format raw testing input
    features = np.concatenate(features_test, axis=0)
    true = labels_test * 1

    # load model with pickle to predict
    with open(pkl_model_path, 'rb') as fin:
        model = pickle.load(fin)

    # Prediction
    preds = model.predict(features)
    scores = np.clip(preds, 1e-15, 1.0 - 1e-15)
    logger('finished prediction')

    # compute auprc
    loss = np.mean(true * -np.log(scores) + (1 - true) * -np.log(1.0 - scores))
    precision, recall, _ = precision_recall_curve(true, scores, pos_label=1)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true, scores, pos_label=1)
    auroc = auc(fpr, tpr)
    # accuracy
    acc = np.sum(true == (scores > 0.5)) / true.shape[0]

    with open("testing_result.pkl", 'wb') as fout:
        pickle.dump((true, scores), fout)

    logger("eval, {}, {}, {}, {}, {}".format(model.num_trees(), loss, auprc, auroc, acc))


def get_datasets(filepaths, is_read_text, limit=None, prefix="", get_label=lambda cols: cols[4] == '9999'):
    def read_bin(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def write_bin(features, labels, filename):
        with open(filename, 'wb') as f:
            pickle.dump((features, labels), f, protocol=4)

    def read_text(filename):
        features = []
        labels = []
        with open(filename) as fread:
            for line in fread:
                cols = line.strip().split()
                if not cols:
                    continue
                cols[29] = DATA_TYPE[cols[29]]
                labels.append(get_label(cols))
                features.append(np.array(
                    [float(cols[i]) for i in range(len(cols)) if i not in removed_features]
                ))
        assert(len(features) == len(labels))
        return (np.array(features), labels)

    interval = 20
    if prefix:
        prefix = prefix + "_"

    filepaths = [filename.strip() for filename in filepaths if filename.strip()]
    removed_features = [0, 1, 3, 4, 5, 7]
    features_list = []
    all_labels = []
    last_pos_features = 0
    last_pos_labels = 0
    for count, filename in enumerate(filepaths):
        bin_filename = prefix + os.path.basename(filename) + ".pkl.data"
        if not is_read_text and not os.path.exists(bin_filename):
            continue
        try:
            if is_read_text:
                features, labels = read_text(filename)
                features_list.append(features)
                all_labels += labels
            else:
                features, labels = read_bin(bin_filename)
                features_list += features
                all_labels    += labels
            logger("loaded " + filename + ", length: " + str(len(features_list)))
        except:
            logger("failed to load " + filename)
        if is_read_text and (count + 1) % interval == 0 and last_pos_features < len(features_list):
            logger("To write {} arrays, {} examples".format(
                len(features_list) - last_pos_features, len(all_labels) - last_pos_labels))
            write_bin(features_list[last_pos_features:], all_labels[last_pos_labels:], bin_filename)
            last_pos_features = len(features_list)
            last_pos_labels = len(all_labels)
        if limit is not None and len(features_list) > limit:
            break
    # Handle last batch
    if is_read_text and last_pos_features < len(features_list):
        write_bin(features_list[last_pos_features:], all_labels[last_pos_labels:], bin_filename)
    # Format labels
    all_labels = np.array(all_labels)
    if np.any(all_labels < 0):
        all_labels = (all_labels > 0) * 1
    print(sum(len(t) for t in features_list), len(features_list), len(all_labels))
    return (features_list, all_labels)


def main_train(config, is_read_text):
    base_dir = config["base_dir"]
    with open(config["training_files"]) as f:
        training_files = f.readlines()
    with open(config["validation_files"]) as f:
        valid_files = f.readlines()

    logger("start constructing datasets")
    (features_train, labels_train) = get_datasets(training_files, is_read_text, prefix="train", limit=3000)
    train = lgb.Dataset(features_train, label=labels_train, params={'max_bin': config["max_bin"]})
    (features_valid, labels_valid) = get_datasets(valid_files, is_read_text, prefix="valid", limit=3000)
    valid = lgb.Dataset(features_valid, label=labels_valid, params={'max_bin': config["max_bin"]})
    logger("start training")
    model = train_lgb(
        train,
        valid,
        config["num_leaves"],
        config["rounds"],
        config["early_stopping_rounds"],
        config["max_bin"],
    )
    logger("finished training")
    (_, pkl_model_path) = persist_model(base_dir, model)
    logger("model is persisted at {}".format(pkl_model_path))


def main_test(config, is_read_text):
    base_dir = config["base_dir"]
    logger("start testing")
    with open(config["testing_files"]) as f:
        testing_files = f.readlines()

    logger("start constructing datasets")
    (features_test, labels_test) = get_datasets(testing_files, is_read_text, prefix="test", limit=3000)

    logger("finished loading testing data")
    pkl_model_path = os.path.join(base_dir, 'model.pkl')
    run_test(features_test, labels_test, pkl_model_path)
    logger("finished testing")


# In[ ]:


DATA_BASE_DIR = "/geosat2/julaiti/tsv_all"
TRAINING_FILES_DESC = os.path.join(DATA_BASE_DIR, "training_files_desc.txt")
VALIDATION_FILES_DESC = os.path.join(DATA_BASE_DIR, "validation_files_desc.txt")
TESTING_FILES_DESC = os.path.join(DATA_BASE_DIR, "testing_files_desc.txt")
config = {
    "base_dir": "./",
    "training_files": TRAINING_FILES_DESC,
    "validation_files": VALIDATION_FILES_DESC,
    "testing_files": TESTING_FILES_DESC,
    "num_leaves": 31,
    "rounds": 1500,
    "early_stopping_rounds": 1500,
    "max_bin": 255,
}


if __name__ == '__main__':
    t0 = time()
    if len(sys.argv) != 3:
        print("Usage: ./lgb.py <train|test|both> <text|bin>")
        sys.exit(1)
    is_ok = False
    if sys.argv[1].lower() in ["both", "train"]:
        if os.path.exists("./train_log_file.txt"):
            print("Train log file exists. Quit.")
            sys.exit(1)
        flog = open("./train_log_file.txt", 'w')
        main_train(config, sys.argv[2].lower() == "text")
        is_ok = True
    if sys.argv[1].lower() in ["both", "test"]:
        if os.path.exists("./test_log_file.txt"):
            print("Test log file exists. Quit.")
            sys.exit(1)
        flog = open("./test_log_file.txt", 'w')
        main_test(config, sys.argv[2].lower() == "text")
        is_ok = True
    if not is_ok:
        print("Cannot understand the parameters.\nUsage: ./lgb.py <train|test|both> <text|bin>")
    flog.close()



# AdaBoost potential function
"""
def expobj(preds, dtrain):
    labels = ((dtrain.get_label() > 0) * 2 - 1).astype("float16")
    hess = np.exp(-labels * preds)  # exp(-margin)
    return -labels * hess, hess     # grad, hess


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# binary error
def exp_eval(preds, data):
    labels = ((data.get_label() > 0) * 2 - 1).astype("float16")
    return 'potential', np.mean(np.exp(-labels * preds)), False


def construct_data(features, labels, max_bin):
    all_train = lgb.Dataset(features, label=labels, params={'max_bin': max_bin})
    all_train.construct()
    size = all_train.num_data()
    # Split training and validating
    thr = int(size * 0.1)
    training = all_train.subset(list(range(0, thr)))
    validating = all_train.subset(list(range(thr, size)))
    # scale pos weight
    labels = all_train.get_label()
    positive = np.array(labels).sum()
    negative = size - positive
    scale_pos_weight = 1.0 * negative / positive
    return ((training, validating), scale_pos_weight)
"""

